#!/usr/bin/env python3
"""
bucket_client.py — variant of client.py using the bucket-based board.

Flow:
 - M1M3 creates a bucket with an encrypted request (no client identifier).
 - M2 polls all buckets, decrypts what it can, processes only buckets targeting it,
   and updates the same bucket payload with the response.
 - M1M3 polls all buckets, decrypts, and when it finds the matching response,
   it acks the bucket to delete it.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    load_mnist,
    batch_accuracy,
    setup_global_logger,
    setup_peer_logger,
)
from models.factory import build_split_models
from clients.bucket_board_client import BucketBoardClient
from peers.base import BaseBucketBoardClient, BaseBucketM1M3Peer, BaseBucketM2Peer, stratified_split


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
    ):
        self.shared_key = shared_key or ""
        logger = setup_peer_logger(name, run_dir, log_level)
        board = BaseBucketBoardClient(board_host, board_port)

        _, M2, _ = build_split_models(
            m1_name=m1_model,
            m2_name=m2_model,
            m3_name=m3_model,
            m2_kwargs={"input_dim": input_dim},
        )
        opt_M2 = optimizers.Adam(learning_rate=lr)

        encode_fn = lambda op, session, sender, tensor: encode_message(
            op,
            session,
            sender,
            tensor,
            self.shared_key,
            target_m2=name,
            pad_multiple=pad_multiple,
        )

        def decode_fn(blob: bytes):
            op, session, sender, tensor, header = decode_message(blob, self.shared_key)
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
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
    ):
        self.shared_key = shared_key or ""
        logger = setup_peer_logger(name, run_dir, log_level)
        board = BaseBucketBoardClient(board_host, board_port)

        M1, _, M3 = build_split_models(
            m1_name=m1_model,
            m2_name=m2_model,
            m3_name=m3_model,
            m3_kwargs={"input_dim": 64},
        )
        opt_M1 = optimizers.Adam(learning_rate=lr)
        opt_M3 = optimizers.Adam(learning_rate=lr)
        loss_fn = losses.SparseCategoricalCrossentropy()

        encode_fn = lambda op, session, sender, tensor: encode_message(
            op,
            session,
            sender_pseudo=None,
            tensor=tensor,
            key_str=self.shared_key,
            target_m2=target_m2,
            pad_multiple=pad_multiple,
        )

        def decode_fn(blob: bytes):
            op, session, sender, tensor, header = decode_message(blob, self.shared_key)
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
            logger=logger,
            verbose=verbose,
            log_every=log_every,
        )

        self.metrics_path = os.path.join(run_dir, f"metrics_{name}.csv")
        with open(self.metrics_path, "w") as f:
            f.write("epoch,step,loss,acc\n")

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
        return acc


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
    suppress_warnings = bool(general.get("suppress_warnings", False))
    log_level = general.get("log_level", "INFO")
    m2_verbose = bool(general.get("m2_verbose", False))
    m2_log_every = int(general.get("m2_log_every", 50))
    m1m3_verbose = bool(general.get("m1m3_verbose", False))
    m1m3_log_every = int(general.get("m1m3_log_every", 50))
    model_arch = general.get("model_architecture", "default")
    m1_model = model_arch
    m2_model = model_arch
    m3_model = model_arch
    if "pad_multiple" not in general:
        raise ValueError("general.pad_multiple must be defined in config.yaml")
    pad_multiple = int(general["pad_multiple"])

    board_host = general.get("board_host", "localhost")
    board_port = int(general.get("board_port", 50051))  # unified board default

    global_logger = setup_global_logger(run_dir, log_level)
    global_logger.info(f"Run dir: {run_dir}")
    global_logger.info(
        f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"m2_verbose={m2_verbose}, m1m3_verbose={m1m3_verbose}, bucket_board_addr={board_host}:{board_port} "
        f"model_architecture={model_arch}"
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

    (x_train, y_train), (x_test, y_test) = load_mnist()
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
            log_level=log_level,
            verbose=m1m3_verbose,
            log_every=m1m3_log_every,
        )
        clients.append(client)
        global_logger.info(f"Spawned Bucket M1M3 peer: {name} → M2: {target_m2} (key_len={len(key)})")

    global_logger.info("Starting training for all bucket-mode clients...")
    ray.get([c.train.remote() for c in clients])

    global_logger.info("Evaluating clients on test set (bucket mode)...")
    accs = ray.get([c.evaluate.remote() for c in clients])
    for c_cfg, acc in zip(m1m3_peers_cfg, accs):
        global_logger.info(f"Client {c_cfg['name']} final test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
