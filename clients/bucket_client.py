#!/usr/bin/env python3
"""
bucket_client.py -- variant of client.py using the bucket-based board.

Flow:
 - M1M3 creates a bucket with an encrypted request (no client identifier).
 - M2 polls all buckets, decrypts what it can, processes only buckets targeting it,
   and updates the same bucket payload with the response.
 - M1M3 polls all buckets, decrypts, and when it finds the matching response,
   it acks the bucket to delete it.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from keras import losses, optimizers
import ray
import yaml

from client import (
    encode_message,
    decode_message,
    load_dataset,
    batch_accuracy,
    setup_global_logger,
    setup_peer_logger,
)
from models.factory import build_split_models
from clients.bucket_board_client import BucketBoardClient
from peers.base import BaseBucketBoardClient, BaseBucketM1M3Peer, BaseBucketM2Peer, BasePIRBoardClient, stratified_split


# ============================================================
# Ray peer: M2 holder with bucket board
# ============================================================

@ray.remote
class BucketPeerM2(BaseBucketM2Peer):
    def __init__(
        self,
        name: str,
        run_dir: str,
        board_host: str,
        board_port: int,
        m1_model: str = "default",
        m2_model: str = "default",
        m3_model: str = "default",
        pad_multiple: int = 1024,
        input_dim: int = 128,
        lr: float = 1e-3,
        shared_key: str | None = None,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
        key_rotation_seconds: int = 0,
        key_rotation_grace: int = 1,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        bucket_namespace: str = "",
        max_messages_per_poll: int = 0,
        batch_delay_ms: int = 0,
        poll_shuffle: bool = False,
        use_pir: bool = False,
        pir_chunk_size: int = 32,
        pir_clue_limit: int = 0,
    ):
        self.shared_key = shared_key or ""
        logger = setup_peer_logger(name, run_dir, log_level)
        board = BaseBucketBoardClient(board_host, board_port)
        pir_client = BasePIRBoardClient(board_host, board_port, logger=logger) if use_pir else None

        _, M2, _ = build_split_models(
            m1_name=m1_model,
            m2_name=m2_model,
            m3_name=m3_model,
            m2_kwargs={"input_dim": input_dim},
        )
        opt_M2 = optimizers.Adam(learning_rate=lr)

        encode_fn = lambda op, session, sender, tensor, reply_token=None: encode_message(
            op,
            session,
            sender,
            tensor,
            self.shared_key,
            target_m2=name,
            pad_multiple=pad_multiple,
            reply_token=reply_token,
            rotation_seconds=key_rotation_seconds,
        )

        def decode_fn(blob: bytes):
            op, session, sender, tensor, header = decode_message(
                blob,
                self.shared_key,
                rotation_seconds=key_rotation_seconds,
                rotation_grace=key_rotation_grace,
            )
            return op, session, sender, tensor, header

        super().__init__(
            name=name,
            board_client=board,
            model=M2,
            optimizer=opt_M2,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            logger=logger,
            target_name=name,
            verbose=verbose,
            log_every=log_every,
            replay_protection=replay_protection,
            replay_cache_size=replay_cache_size,
            replay_window_ms=replay_window_ms,
            replay_future_ms=replay_future_ms,
            send_jitter_ms=send_jitter_ms,
            poll_jitter_ms=poll_jitter_ms,
            dummy_rate=dummy_rate,
            bucket_namespace=bucket_namespace,
            max_messages_per_poll=max_messages_per_poll,
            batch_delay_ms=batch_delay_ms,
            poll_shuffle=poll_shuffle,
            use_pir=use_pir,
            pir_client=pir_client,
            pir_key=self.shared_key,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
            pir_rotation_seconds=key_rotation_seconds,
            pir_rotation_grace=key_rotation_grace,
        )

        logger.info(
            "Initialized BucketPeerM2 input_dim=%d lr=%.4f board=%s:%d shared_key_len=%d",
            input_dim,
            lr,
            board_host,
            board_port,
            len(self.shared_key),
        )


# ============================================================
# Ray peer: M1+M3 client with bucket board
# ============================================================

@ray.remote
class BucketPeerM1M3(BaseBucketM1M3Peer):
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
        target_m2: str,
        m1_model: str = "default",
        m2_model: str = "default",
        m3_model: str = "default",
        pad_multiple: int = 1024,
        shared_key: str | None = None,
        epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        max_steps_per_epoch: int = 0,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
        key_rotation_seconds: int = 0,
        key_rotation_grace: int = 1,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        bucket_namespace: str = "",
        pseudonym_scope: str = "per_run",
        poll_shuffle: bool = False,
        hide_sender: bool = False,
        use_pir: bool = False,
        pir_chunk_size: int = 32,
        pir_clue_limit: int = 0,
    ):
        self.shared_key = shared_key or ""
        self.hide_sender = bool(hide_sender)
        logger = setup_peer_logger(name, run_dir, log_level)
        board = BaseBucketBoardClient(board_host, board_port)
        pir_client = BasePIRBoardClient(board_host, board_port, logger=logger) if use_pir else None

        M1, _, M3 = build_split_models(
            m1_name=m1_model,
            m2_name=m2_model,
            m3_name=m3_model,
            m3_kwargs={"input_dim": 64},
        )
        opt_M1 = optimizers.Adam(learning_rate=lr)
        opt_M3 = optimizers.Adam(learning_rate=lr)
        loss_fn = losses.SparseCategoricalCrossentropy()

        def encode_fn(op, session, sender, tensor, reply_token=None):
            sender_pseudo = None if hide_sender else sender
            return encode_message(
            op,
            session,
            sender_pseudo=sender_pseudo,
            tensor=tensor,
            key_str=self.shared_key,
            target_m2=target_m2,
            pad_multiple=pad_multiple,
            reply_token=reply_token,
            rotation_seconds=key_rotation_seconds,
            )

        def decode_fn(blob: bytes):
            op, session, sender, tensor, header = decode_message(
                blob,
                self.shared_key,
                rotation_seconds=key_rotation_seconds,
                rotation_grace=key_rotation_grace,
            )
            return op, session, sender, tensor, header

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
            target_m2=target_m2,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
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
            bucket_namespace=bucket_namespace,
            pseudonym_scope=pseudonym_scope,
            poll_shuffle=poll_shuffle,
            use_pir=use_pir,
            pir_client=pir_client,
            pir_key=self.shared_key,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
            pir_rotation_seconds=key_rotation_seconds,
            pir_rotation_grace=key_rotation_grace,
        )

        self.metrics_path = os.path.join(run_dir, f"metrics_{name}.csv")
        with open(self.metrics_path, "w") as f:
            f.write("epoch,step,loss,acc\n")
        self._last_acc = None

        logger.info(
            "Initialized BucketPeerM1M3 name=%s target_m2=%s epochs=%d batch_size=%d lr=%.4f board=%s:%d shared_key_len=%d models(m1/m2/m3)=%s/%s/%s",
            name,
            target_m2,
            epochs,
            batch_size,
            lr,
            board_host,
            board_port,
            len(self.shared_key),
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
            "Summary name=%s target_m2=%s pseudonym_scope=%s hide_sender=%s bytes_sent=%d bytes_received=%d acc=%.4f",
            self.name,
            self.target_m2,
            self.pseudonym_scope,
            self.hide_sender,
            self.bytes_sent,
            self.bytes_received,
            self._last_acc,
        )
        return acc

    def summary(self) -> dict:
        return {
            "name": self.name,
            "sender_id": "",
            "target_m2": self.target_m2,
            "pseudonym_scope": self.pseudonym_scope,
            "hide_sender": self.hide_sender,
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
    m2_verbose = bool(general.get("m2_verbose", False))
    m2_log_every = int(general.get("m2_log_every", 50))
    m1m3_verbose = bool(general.get("m1m3_verbose", False))
    m1m3_log_every = int(general.get("m1m3_log_every", 50))
    model_arch = general.get("model_architecture", "default")
    dataset = general.get("dataset", "mnist")
    m1_model = model_arch
    m2_model = model_arch
    m3_model = model_arch
    if "pad_multiple" not in general:
        raise ValueError("general.pad_multiple must be defined in config.yaml")
    pad_multiple = int(general["pad_multiple"])

    replay_protection = bool(general.get("replay_protection", True))
    replay_cache_size = int(general.get("replay_cache_size", 10000))
    replay_window_ms = int(general.get("replay_window_ms", 300000))
    replay_future_ms = int(general.get("replay_future_ms", 60000))
    send_jitter_ms = int(general.get("send_jitter_ms", 0))
    poll_jitter_ms = int(general.get("poll_jitter_ms", 0))
    dummy_rate = float(general.get("dummy_message_rate", 0.0))
    key_rotation_seconds = int(general.get("key_rotation_seconds", 0))
    key_rotation_grace = int(general.get("key_rotation_grace", 1))
    pseudonym_scope = general.get("pseudonym_scope", "per_session")
    bucket_namespace_mode = general.get("bucket_namespace_mode", "global")
    m2_max_messages_per_poll = int(general.get("m2_max_messages_per_poll", 0))
    m2_batch_delay_ms = int(general.get("m2_batch_delay_ms", 0))
    poll_shuffle = bool(general.get("poll_shuffle", False))
    hide_sender = bool(general.get("hide_sender", False))
    use_pir = bool(general.get("use_pir", False))
    pir_chunk_size = int(general.get("pir_chunk_size", 32))
    pir_clue_limit = int(general.get("pir_clue_limit", 0))

    board_host = general.get("board_host", "localhost")
    board_port = int(general.get("board_port", 50051))  # unified board default

    global_logger = setup_global_logger(run_dir, log_level)
    global_logger.info(f"Run dir: {run_dir}")
    global_logger.info(
        f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"m2_verbose={m2_verbose}, m1m3_verbose={m1m3_verbose}, bucket_board_addr={board_host}:{board_port} "
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

    m2_keys: Dict[str, str] = {}
    for m2_cfg in m2_peers_cfg:
        name = m2_cfg["name"]
        key = m2_cfg.get("key", "") or ""
        m2_keys[name] = key

    global_logger.info(f"Using bucket board at {board_host}:{board_port}")

    x_shards, y_shards = stratified_split(x_train, y_train, n_clients, seed=seed)

    m2_peers: Dict[str, Any] = {}
    for m2_cfg in m2_peers_cfg:
        name = m2_cfg["name"]
        key = m2_keys.get(name, "")
        m2_namespace = name if bucket_namespace_mode == "per_m2" else ""
        m2_peer = BucketPeerM2.remote(
            name=name,
            run_dir=run_dir,
            board_host=board_host,
            board_port=board_port,
            m1_model=m1_model,
            m2_model=m2_model,
            m3_model=m3_model,
            input_dim=128,
            lr=lr,
            shared_key=key,
            log_level=log_level,
            verbose=m2_verbose,
            log_every=m2_log_every,
            key_rotation_seconds=key_rotation_seconds,
            key_rotation_grace=key_rotation_grace,
            replay_protection=replay_protection,
            replay_cache_size=replay_cache_size,
            replay_window_ms=replay_window_ms,
            replay_future_ms=replay_future_ms,
            send_jitter_ms=send_jitter_ms,
            poll_jitter_ms=poll_jitter_ms,
            dummy_rate=dummy_rate,
            bucket_namespace=m2_namespace,
            max_messages_per_poll=m2_max_messages_per_poll,
            batch_delay_ms=m2_batch_delay_ms,
            poll_shuffle=poll_shuffle,
            use_pir=use_pir,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
        )
        m2_peers[name] = m2_peer
        global_logger.info(f"Spawned Bucket M2 peer: {name} (key_len={len(key)})")

    for name, m2_peer in m2_peers.items():
        m2_peer.run.remote()
        global_logger.info(f"Started run() loop for Bucket M2 peer: {name}")

    clients = []
    for i, c_cfg in enumerate(m1m3_peers_cfg):
        name = c_cfg["name"]
        target_m2 = c_cfg["target_m2"]
        if target_m2 not in m2_peers:
            raise ValueError(f"M1M3 peer {name} references unknown M2 peer '{target_m2}'")
        key = m2_keys.get(target_m2, "")
        client_namespace = target_m2 if bucket_namespace_mode == "per_m2" else ""
        client = BucketPeerM1M3.remote(
            name=name,
            run_dir=run_dir,
            x_train=x_shards[i],
            y_train=y_shards[i],
            x_test=x_test,
            y_test=y_test,
            board_host=board_host,
            board_port=board_port,
            m1_model=m1_model,
            m2_model=m2_model,
            m3_model=m3_model,
            target_m2=target_m2,
            shared_key=key,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_steps_per_epoch=max_steps_per_epoch,
            log_level=log_level,
            verbose=m1m3_verbose,
            log_every=m1m3_log_every,
            key_rotation_seconds=key_rotation_seconds,
            key_rotation_grace=key_rotation_grace,
            replay_protection=replay_protection,
            replay_cache_size=replay_cache_size,
            replay_window_ms=replay_window_ms,
            replay_future_ms=replay_future_ms,
            send_jitter_ms=send_jitter_ms,
            poll_jitter_ms=poll_jitter_ms,
            dummy_rate=dummy_rate,
            bucket_namespace=client_namespace,
            pseudonym_scope=pseudonym_scope,
            poll_shuffle=poll_shuffle,
            hide_sender=hide_sender,
            use_pir=use_pir,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
        )
        clients.append(client)
        global_logger.info(f"Spawned Bucket M1M3 peer: {name} -> M2: {target_m2} (key_len={len(key)})")

    global_logger.info("Starting training for all bucket-mode clients...")
    ray.get([c.train.remote() for c in clients])

    global_logger.info("Evaluating clients on test set (bucket mode)...")
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
