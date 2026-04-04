#!/usr/bin/env python3
"""
vanilla_split.py -- baseline split learning with Ray actors and plaintext Board messages.

This uses the shared BaseBoardClient/BaseM*Peer classes; privacy features (encryption,
pseudonyms, padding) are intentionally absent here.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any

import numpy as np
import ray
import yaml
from keras import losses, optimizers

from client import load_dataset, batch_accuracy, setup_global_logger, setup_peer_logger
from models.factory import build_split_models
from peers.base import BaseBoardClient, BaseM1M3Peer, BaseM2Peer, stratified_split


# ============================================================
# Plain serialization helpers (no encryption)
# ============================================================


def tensor_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def bytes_to_tensor(b: bytes) -> np.ndarray:
    buf = io.BytesIO(b)
    return np.load(buf, allow_pickle=False)


def encode_plain_message(
    op: str,
    session: str,
    sender: str,
    tensor: np.ndarray,
    reply_token: str | None = None,
    perf: dict | None = None,
) -> bytes:
    """Build a plaintext envelope: 4-byte header_len || header_json || tensor_bytes."""
    t0 = time.perf_counter()
    tensor_bytes = tensor_to_bytes(tensor)
    header = {
        "op": op,
        "session": session,
        "msg_id": uuid.uuid4().hex,
        "ts_ms": int(time.time() * 1000),
        "sender": sender,
        "tensor_len": len(tensor_bytes),
    }
    if reply_token:
        header["reply_token"] = reply_token
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_bytes).to_bytes(4, "big")
    blob = header_len + header_bytes + tensor_bytes
    t1 = time.perf_counter()
    if perf is not None:
        perf["serialize_ms"] = (t1 - t0) * 1000.0
        perf["encrypt_ms"] = 0.0
        perf["total_ms"] = (t1 - t0) * 1000.0
        perf["header_bytes"] = len(header_bytes)
        perf["tensor_bytes"] = len(tensor_bytes)
        perf["pad_bytes"] = 0
        perf["crypto_overhead_bytes"] = 0
        perf["total_bytes"] = len(blob)
    return blob


def decode_plain_message(blob: bytes, perf: dict | None = None):
    """Inverse of encode_plain_message, returns (op, session, sender, tensor, header)."""
    t0 = time.perf_counter()
    if len(blob) < 4:
        raise ValueError("payload too short for header length")
    hlen = int.from_bytes(blob[:4], "big")
    if len(blob) < 4 + hlen:
        raise ValueError("payload truncated before header")
    header = json.loads(blob[4:4 + hlen].decode("utf-8"))
    tlen = int(header["tensor_len"])
    start = 4 + hlen
    end = start + tlen
    if len(blob) < end:
        raise ValueError("payload truncated before tensor")
    tensor = bytes_to_tensor(blob[start:end])
    t1 = time.perf_counter()
    if perf is not None:
        perf["decrypt_ms"] = 0.0
        perf["deserialize_ms"] = (t1 - t0) * 1000.0
        perf["total_ms"] = (t1 - t0) * 1000.0
        perf["header_bytes"] = int(hlen)
        perf["tensor_bytes"] = int(tlen)
        perf["pad_bytes"] = 0
        perf["crypto_overhead_bytes"] = 0
        perf["total_bytes"] = len(blob)
        header["_perf"] = perf
    return header["op"], header["session"], header.get("sender", ""), tensor, header


# ============================================================
# Ray peers
# ============================================================


@ray.remote
class VanillaPeerM2(BaseM2Peer):
    def __init__(
        self,
        name: str,
        run_dir: str,
        board_host: str,
        board_port: int,
        m1_model: str = "default",
        m2_model: str = "default",
        m3_model: str = "default",
        input_dim: int = 128,
        lr: float = 1e-3,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        max_messages_per_poll: int = 1,
        batch_delay_ms: int = 0,
        perf_path: str | None = None,
    ):
        logger = setup_peer_logger(name, run_dir, log_level)
        board = BaseBoardClient(board_host, board_port, logger=logger)

        _, M2, _ = build_split_models(
            m1_name=m1_model,
            m2_name=m2_model,
            m3_name=m3_model,
            m2_kwargs={"input_dim": input_dim},
        )
        opt_M2 = optimizers.Adam(learning_rate=lr)

        encode_fn = lambda op, session, sender, tensor, reply_token=None, perf=None: encode_plain_message(
            op,
            session,
            sender,
            tensor,
            reply_token=reply_token,
            perf=perf,
        )
        decode_fn = lambda blob, perf=None: decode_plain_message(blob, perf=perf)

        super().__init__(
            name=name,
            board_client=board,
            model=M2,
            optimizer=opt_M2,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            logger=logger,
            verbose=verbose,
            log_every=log_every,
            replay_protection=replay_protection,
            replay_cache_size=replay_cache_size,
            replay_window_ms=replay_window_ms,
            replay_future_ms=replay_future_ms,
            send_jitter_ms=send_jitter_ms,
            poll_jitter_ms=poll_jitter_ms,
            dummy_rate=dummy_rate,
            max_messages_per_poll=max_messages_per_poll,
            batch_delay_ms=batch_delay_ms,
            perf_path=perf_path,
        )

        logger.info(
            "Initialized VanillaPeerM2 input_dim=%d lr=%.4f board=%s:%d models(m1/m2/m3)=%s/%s/%s",
            input_dim,
            lr,
            board_host,
            board_port,
            m1_model,
            m2_model,
            m3_model,
        )


@ray.remote
class VanillaPeerM1M3(BaseM1M3Peer):
    def __init__(
        self,
        name: str,
        run_dir: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        board_host: str,
        board_port: int,
        m2_name: str,
        m1_model: str = "default",
        m2_model: str = "default",
        m3_model: str = "default",
        epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        max_steps_per_epoch: int = 0,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        perf_path: str | None = None,
    ):
        logger = setup_peer_logger(name, run_dir, log_level)
        board = BaseBoardClient(board_host, board_port, logger=logger)

        M1, _, M3 = build_split_models(
            m1_name=m1_model,
            m2_name=m2_model,
            m3_name=m3_model,
            m3_kwargs={"input_dim": 64},
        )
        opt_M1 = optimizers.Adam(learning_rate=lr)
        opt_M3 = optimizers.Adam(learning_rate=lr)
        loss_fn = losses.SparseCategoricalCrossentropy()

        encode_fn = lambda op, session, sender, tensor, reply_token=None: encode_plain_message(
            op,
            session,
            sender,
            tensor,
            reply_token=reply_token,
        )
        decode_fn = lambda blob: decode_plain_message(blob)

        super().__init__(
            name=name,
            run_dir=run_dir,
            board_client=board,
            M1=M1,
            M3=M3,
            opt_M1=opt_M1,
            opt_M3=opt_M3,
            loss_fn=loss_fn,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            target_m2=m2_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            sender_id=name,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            max_steps_per_epoch=max_steps_per_epoch,
            logger=logger,
            verbose=verbose,
            log_every=log_every,
            replay_protection=replay_protection,
            replay_cache_size=replay_cache_size,
            replay_window_ms=replay_window_ms,
            replay_future_ms=replay_future_ms,
            send_jitter_ms=send_jitter_ms,
            poll_jitter_ms=poll_jitter_ms,
            dummy_rate=dummy_rate,
            perf_path=perf_path,
        )

        self.metrics_path = os.path.join(run_dir, f"metrics_{name}.csv")
        with open(self.metrics_path, "w") as f:
            f.write("epoch,step,loss,acc\n")
        self._last_acc = None

        logger.info(
            "Initialized VanillaPeerM1M3 name=%s target_m2=%s epochs=%d batch_size=%d lr=%.4f board=%s:%d models(m1/m2/m3)=%s/%s/%s",
            name,
            m2_name,
            epochs,
            batch_size,
            lr,
            board_host,
            board_port,
            m1_model,
            m2_model,
            m3_model,
        )

    def train(self):
        def write_metrics(epoch: int, step: int, loss_val: float, acc: float):
            with open(self.metrics_path, "a") as f:
                f.write(f"{epoch},{step},{loss_val},{acc}\n")

        self.train_batches(write_metrics)
        return f"{self.name} training finished."

    def evaluate(self) -> float:
        acc = self.evaluate_batches()
        self.logger.info("Test accuracy: %.4f", acc)
        self._last_acc = float(acc)
        self.logger.info(
            "Summary name=%s sender_id=%s target_m2=%s bytes_sent=%d bytes_received=%d acc=%.4f",
            self.name,
            self.sender_id,
            self.target_m2,
            self.bytes_sent,
            self.bytes_received,
            self._last_acc,
        )
        return acc

    def summary(self) -> dict:
        return {
            "name": self.name,
            "sender_id": self.sender_id,
            "target_m2": self.target_m2,
            "pseudonym_scope": "none",
            "hide_sender": False,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "last_acc": self._last_acc,
        }


# ============================================================
# Main
# ============================================================


def main(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_cfg = cfg.get("run", {})
    general = cfg.get("general", {})
    peers_cfg = cfg.get("peers", {})

    base_dir = run_cfg.get("base_dir", "runs")
    run_name = run_cfg.get("name") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    epochs = int(general.get("epochs", 2))
    batch_size = int(general.get("batch_size", 128))
    lr = float(general.get("lr", 1e-3))
    max_steps_per_epoch = int(general.get("max_steps_per_epoch", 0))
    suppress_warnings = bool(general.get("suppress_warnings", False))
    log_level = general.get("log_level", "INFO")
    model_arch = general.get("model_architecture", "default")
    dataset = general.get("dataset", "mnist")
    m1_model = model_arch
    m2_model = model_arch
    m3_model = model_arch

    m2_verbose = bool(general.get("m2_verbose", False))
    m2_log_every = int(general.get("m2_log_every", 50))
    m1m3_verbose = bool(general.get("m1m3_verbose", False))
    m1m3_log_every = int(general.get("m1m3_log_every", 50))

    replay_protection = bool(general.get("replay_protection", True))
    replay_cache_size = int(general.get("replay_cache_size", 10000))
    replay_window_ms = int(general.get("replay_window_ms", 300000))
    replay_future_ms = int(general.get("replay_future_ms", 60000))
    send_jitter_ms = int(general.get("send_jitter_ms", 0))
    poll_jitter_ms = int(general.get("poll_jitter_ms", 0))
    dummy_rate = float(general.get("dummy_message_rate", 0.0))
    m2_max_messages_per_poll = int(general.get("m2_max_messages_per_poll", 1))
    m2_batch_delay_ms = int(general.get("m2_batch_delay_ms", 0))
    enable_perf = bool(general.get("enable_perf_metrics", False))

    board_host = general.get("board_host", "localhost")
    board_port = int(general.get("board_port", 50051))

    global_logger = setup_global_logger(run_dir, log_level)
    global_logger.info(f"Run dir: {run_dir}")
    global_logger.info(
        f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"m2_verbose={m2_verbose}, m1m3_verbose={m1m3_verbose}, board_addr={board_host}:{board_port}, "
        f"model_architecture={model_arch}, dataset={dataset}"
    )
    try:
        import shutil
        shutil.copyfile(config_path, os.path.join(run_dir, "config_used.yaml"))
    except Exception as e:
        global_logger.warning(f"Could not save config snapshot: {e}")

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray_logging_level = logging.ERROR if suppress_warnings else logging.INFO
    ray_log_to_driver = not suppress_warnings
    ray.init(ignore_reinit_error=True, logging_level=ray_logging_level, log_to_driver=ray_log_to_driver)

    (x_train, y_train), (x_test, y_test) = load_dataset(dataset)
    if "random_seed" not in general:
        raise ValueError("general.random_seed must be defined in config.yaml")
    seed = int(general.get("random_seed", 42))

    m1m3_peers_cfg = peers_cfg.get("M1M3", [])
    m2_peers_cfg = peers_cfg.get("M2", [])
    if not m1m3_peers_cfg or not m2_peers_cfg:
        raise ValueError("Config must define at least one M1M3 peer and one M2 peer.")
    n_clients = len(m1m3_peers_cfg)
    global_logger.info(f"Configured {n_clients} M1M3 peers and {len(m2_peers_cfg)} M2 peers.")

    x_shards, y_shards = stratified_split(x_train, y_train, n_clients, seed=seed)

    m2_peers: Dict[str, Any] = {}
    for m2_cfg in m2_peers_cfg:
        name = m2_cfg["name"]
        m2_peer = VanillaPeerM2.remote(
            name=name,
            run_dir=run_dir,
            board_host=board_host,
            board_port=board_port,
            m1_model=m1_model,
            m2_model=m2_model,
            m3_model=m3_model,
            input_dim=128,
            lr=lr,
            log_level=log_level,
            verbose=m2_verbose,
            log_every=m2_log_every,
            replay_protection=replay_protection,
            replay_cache_size=replay_cache_size,
            replay_window_ms=replay_window_ms,
            replay_future_ms=replay_future_ms,
            send_jitter_ms=send_jitter_ms,
            poll_jitter_ms=poll_jitter_ms,
            dummy_rate=dummy_rate,
            max_messages_per_poll=m2_max_messages_per_poll,
            batch_delay_ms=m2_batch_delay_ms,
            perf_path=os.path.join(run_dir, f"perf_m2_{name}.csv") if enable_perf else None,
        )
        m2_peers[name] = m2_peer
        global_logger.info(f"Spawned Vanilla M2 peer: {name}")

    for name, m2_peer in m2_peers.items():
        m2_peer.run.remote()
        global_logger.info(f"Started run() loop for Vanilla M2 peer: {name}")

    clients = []
    for i, c_cfg in enumerate(m1m3_peers_cfg):
        name = c_cfg["name"]
        target_m2 = c_cfg["target_m2"]
        if target_m2 not in m2_peers:
            raise ValueError(f"M1M3 peer {name} references unknown M2 peer '{target_m2}'")
        client = VanillaPeerM1M3.remote(
            name=name,
            run_dir=run_dir,
            x_train=x_shards[i],
            y_train=y_shards[i],
            x_test=x_test,
            y_test=y_test,
            board_host=board_host,
            board_port=board_port,
            m2_name=target_m2,
            m1_model=m1_model,
            m2_model=m2_model,
            m3_model=m3_model,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_steps_per_epoch=max_steps_per_epoch,
            log_level=log_level,
            verbose=m1m3_verbose,
            log_every=m1m3_log_every,
            replay_protection=replay_protection,
            replay_cache_size=replay_cache_size,
            replay_window_ms=replay_window_ms,
            replay_future_ms=replay_future_ms,
            send_jitter_ms=send_jitter_ms,
            poll_jitter_ms=poll_jitter_ms,
            dummy_rate=dummy_rate,
            perf_path=os.path.join(run_dir, f"perf_steps_{name}.csv") if enable_perf else None,
        )
        clients.append(client)
        global_logger.info(f"Spawned Vanilla M1M3 peer: {name} -> M2: {target_m2}")

    global_logger.info("Starting training for all vanilla clients...")
    ray.get([c.train.remote() for c in clients])

    global_logger.info("Evaluating clients on test set (vanilla)...")
    accs = ray.get([c.evaluate.remote() for c in clients])
    for c_cfg, acc in zip(m1m3_peers_cfg, accs):
        global_logger.info(f"Client {c_cfg['name']} final test accuracy: {acc:.4f}")

    summaries = ray.get([c.summary.remote() for c in clients])
    for summary in summaries:
        global_logger.info(
            "Client summary name=%s sender_id=%s target_m2=%s pseudonym_scope=%s hide_sender=%s bytes_sent=%d bytes_received=%d acc=%s",
            summary.get("name"),
            summary.get("sender_id"),
            summary.get("target_m2"),
            summary.get("pseudonym_scope"),
            summary.get("hide_sender"),
            summary.get("bytes_sent"),
            summary.get("bytes_received"),
            summary.get("last_acc"),
        )


if __name__ == "__main__":
    main()
