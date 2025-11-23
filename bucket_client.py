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
    AUDIENCE_TO_M2,
    AUDIENCE_TO_CLIENTS,
)
from models.factory import build_split_models
from bucket_board_client import BucketBoardClient


# ============================================================
# Ray peer: M2 holder with bucket board
# ============================================================

@ray.remote
class BucketPeerM2:
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
        shared_key: str | None = None,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
    ):
        self.name = name
        self.logger = setup_peer_logger(name, run_dir, log_level)
        self.board = BucketBoardClient(board_host, board_port)
        self.m1_model = m1_model
        self.m2_model = m2_model
        self.m3_model = m3_model

        _, self.M2, _ = build_split_models(
            m1_name=self.m1_model,
            m2_name=self.m2_model,
            m3_name=self.m3_model,
            m2_kwargs={"input_dim": input_dim},
        )
        self.opt_M2 = optimizers.Adam(learning_rate=lr)
        self._sessions: Dict[str, tuple] = {}
        self._seen_bucket_ops: Dict[str, str] = {}

        self.shared_key = shared_key or ""
        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0

        self.logger.info(
            f"Initialized BucketPeerM2 input_dim={input_dim} lr={lr} board={board_host}:{board_port} "
            f"shared_key_len={len(self.shared_key)}"
        )

    def run(self):
        self.logger.info("BucketPeerM2.run loop started.")
        while True:
            buckets = self.board.poll_buckets()
            if not buckets:
                time.sleep(0.01)
                continue

            for b in buckets:
                bid = b["bucket_id"]
                payload = b["payload"]

                try:
                    op, session, sender, tensor, header = decode_message(payload, self.shared_key)
                except Exception:
                    # not for this M2/key
                    continue

                target = header.get("target_m2")
                if target and target != self.name:
                    continue

                # Remember last seen op for this bucket
                self._seen_bucket_ops[bid] = op

                if op == "FWD_REQ":
                    self._handle_forward(bid, session, tensor)
                elif op == "BWD_REQ":
                    self._handle_backward(bid, session, tensor)
                elif op == "INFER_REQ":
                    self._handle_infer(bid, session, tensor)
                else:
                    continue

    def _handle_forward(self, bucket_id: str, session_id: str, z_cut_np: np.ndarray):
        self._fwd_count += 1
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z_cut)
            z_mid = self.M2(z_cut, training=True)
        self._sessions[session_id] = (tape, z_cut, z_mid)
        z_mid_np = z_mid.numpy()

        resp = encode_message("FWD_RES", session_id, self.name, z_mid_np, self.shared_key, target_m2=self.name)
        self.board.update_bucket(bucket_id, resp)
        self._seen_bucket_ops[bucket_id] = "FWD_RES"

    def _handle_backward(self, bucket_id: str, session_id: str, dL_dz_mid_np: np.ndarray):
        self._bwd_count += 1
        if session_id not in self._sessions:
            self.logger.error(f"[BWD] no cached forward for session={session_id}")
            return
        tape, z_cut, z_mid = self._sessions.pop(session_id)
        dL_dz_mid = tf.convert_to_tensor(dL_dz_mid_np, dtype=tf.float32)
        targets = self.M2.trainable_variables + [z_cut]
        grads_all = tape.gradient(z_mid, targets, output_gradients=dL_dz_mid)
        grads_M2 = grads_all[:-1]
        dL_dz_cut = grads_all[-1]
        self.opt_M2.apply_gradients(zip(grads_M2, self.M2.trainable_variables))
        dL_dz_cut_np = dL_dz_cut.numpy()

        resp = encode_message("BWD_RES", session_id, self.name, dL_dz_cut_np, self.shared_key, target_m2=self.name)
        self.board.update_bucket(bucket_id, resp)
        self._seen_bucket_ops[bucket_id] = "BWD_RES"

    def _handle_infer(self, bucket_id: str, session_id: str, z_cut_np: np.ndarray):
        self._infer_count += 1
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        z_mid = self.M2(z_cut, training=False)
        z_mid_np = z_mid.numpy()

        resp = encode_message("INFER_RES", session_id, self.name, z_mid_np, self.shared_key, target_m2=self.name)
        self.board.update_bucket(bucket_id, resp)
        self._seen_bucket_ops[bucket_id] = "INFER_RES"


# ============================================================
# Ray peer: M1+M3 client with bucket board
# ============================================================

@ray.remote
class BucketPeerM1M3:
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
        shared_key: str | None = None,
        epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
    ):
        self.actor_name = name
        self.logger = setup_peer_logger(name, run_dir, log_level)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.board = BucketBoardClient(board_host, board_port)
        self.target_m2 = target_m2
        self.m1_model = m1_model
        self.m2_model = m2_model
        self.m3_model = m3_model

        self.epochs = epochs
        self.batch_size = batch_size

        self.M1, _, self.M3 = build_split_models(
            m1_name=self.m1_model,
            m2_name=self.m2_model,
            m3_name=self.m3_model,
            m3_kwargs={"input_dim": 64},
        )
        self.opt_M1 = optimizers.Adam(learning_rate=lr)
        self.opt_M3 = optimizers.Adam(learning_rate=lr)
        self.loss_fn = losses.SparseCategoricalCrossentropy()

        self.shared_key = shared_key or ""
        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0
        self._seen_buckets: set[str] = set()

        self.pseudonym = f"cli_{uuid.uuid4().hex[:8]}"

        self.metrics_path = os.path.join(run_dir, f"metrics_{self.actor_name}.csv")
        with open(self.metrics_path, "w") as f:
            f.write("epoch,step,loss,acc\n")

        self.logger.info(
            f"Initialized BucketPeerM1M3 actor_name={self.actor_name} pseudonym={self.pseudonym} "
            f"epochs={epochs} batch_size={batch_size} lr={lr} target_m2={target_m2} "
            f"board={board_host}:{board_port} shared_key_len={len(self.shared_key)} "
            f"models(m1/m2/m3)={self.m1_model}/{self.m2_model}/{self.m3_model}"
        )

    def train(self):
        n = len(self.x_train)
        steps_per_epoch = n // self.batch_size
        self.logger.info(f"Starting training on {n} samples, {steps_per_epoch} steps/epoch.")

        for epoch in range(1, self.epochs + 1):
            idx = np.random.permutation(n)
            x_sh = self.x_train[idx]
            y_sh = self.y_train[idx]

            epoch_losses, epoch_accs = [], []
            start = time.time()

            for step in range(steps_per_epoch):
                lo = step * self.batch_size
                hi = lo + self.batch_size
                xb = tf.convert_to_tensor(x_sh[lo:hi], dtype=tf.float32)
                yb = tf.convert_to_tensor(y_sh[lo:hi], dtype=tf.int32)

                with tf.GradientTape(persistent=True) as tape_M1:
                    z_cut = self.M1(xb, training=True)
                z_cut_np = z_cut.numpy()

                session_id = f"{self.actor_name}-train-{epoch}-{step}-{uuid.uuid4().hex}"

                # Create bucket with FWD_REQ
                self._fwd_count += 1
                payload = encode_message(
                    "FWD_REQ",
                    session_id,
                    sender_pseudo=None,
                    tensor=z_cut_np,
                    key_str=self.shared_key,
                    target_m2=self.target_m2,
                )
                bucket_id = self.board.create_bucket(payload)

                # Wait for FWD_RES in bucket
                z_mid_np = self._wait_for_response(bucket_id, session_id, "FWD_RES")
                z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)

                # M3 forward/backward
                with tf.GradientTape() as tape_M3:
                    tape_M3.watch(z_mid)
                    logits = self.M3(z_mid, training=True)
                    loss_value = self.loss_fn(yb, logits)

                targets = self.M3.trainable_variables + [z_mid]
                grads_all = tape_M3.gradient(loss_value, targets)
                grads_M3 = grads_all[:-1]
                dL_dz_mid = grads_all[-1]
                self.opt_M3.apply_gradients(zip(grads_M3, self.M3.trainable_variables))

                # Update bucket with BWD_REQ
                self._bwd_count += 1
                dL_dz_mid_np = dL_dz_mid.numpy()
                payload = encode_message(
                    "BWD_REQ",
                    session_id,
                    sender_pseudo=None,
                    tensor=dL_dz_mid_np,
                    key_str=self.shared_key,
                    target_m2=self.target_m2,
                )
                self.board.update_bucket(bucket_id, payload)

                # Wait for BWD_RES then ack bucket
                dL_dz_cut_np = self._wait_for_response(bucket_id, session_id, "BWD_RES")
                self.board.ack_bucket(bucket_id)

                dL_dz_cut = tf.convert_to_tensor(dL_dz_cut_np, dtype=tf.float32)
                grads_M1 = tape_M1.gradient(z_cut, self.M1.trainable_variables, output_gradients=dL_dz_cut)
                self.opt_M1.apply_gradients(zip(grads_M1, self.M1.trainable_variables))
                del tape_M1

                acc_batch = batch_accuracy(yb.numpy(), logits.numpy())
                loss_val = float(loss_value.numpy())
                epoch_losses.append(loss_val)
                epoch_accs.append(acc_batch)
                with open(self.metrics_path, "a") as f:
                    f.write(f"{epoch},{step},{loss_val},{acc_batch}\n")

                if step % 100 == 0:
                    self.logger.info(
                        f"Epoch {epoch} Step {step}/{steps_per_epoch} "
                        f"Loss={loss_val:.4f} Acc={acc_batch:.4f}"
                    )

            mean_loss = float(np.mean(epoch_losses))
            mean_acc = float(np.mean(epoch_accs))
            elapsed = time.time() - start
            self.logger.info(f"Epoch {epoch} done in {elapsed:.1f}s → Loss={mean_loss:.4f} Acc={mean_acc:.4f}")

        return f"{self.actor_name} training finished."

    def _wait_for_response(self, bucket_id: str, session_id: str, expect_op: str):
        while True:
            buckets = self.board.poll_buckets()
            if not buckets:
                time.sleep(0.01)
                continue
            for b in buckets:
                if b["bucket_id"] != bucket_id:
                    continue
                try:
                    op, sess, sender, tensor, header = decode_message(b["payload"], self.shared_key)
                except Exception:
                    continue
                if op == expect_op and sess == session_id:
                    return tensor
            time.sleep(0.01)

    def evaluate(self) -> float:
        batch_size = self.batch_size
        n = len(self.x_test)
        all_logits = []

        for i in range(0, n, batch_size):
            xb = tf.convert_to_tensor(self.x_test[i:i + batch_size], dtype=tf.float32)
            z_cut = self.M1(xb, training=False)
            z_cut_np = z_cut.numpy()

            session_id = f"{self.actor_name}-eval-{i}-{uuid.uuid4().hex}"

            payload = encode_message(
                "INFER_REQ",
                session_id,
                sender_pseudo=None,
                tensor=z_cut_np,
                key_str=self.shared_key,
                target_m2=self.target_m2,
            )
            bucket_id = self.board.create_bucket(payload)

            z_mid_np = self._wait_for_response(bucket_id, session_id, "INFER_RES")
            self.board.ack_bucket(bucket_id)

            z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)
            logits = self.M3(z_mid, training=False)
            all_logits.append(logits.numpy())

        logits_full = np.concatenate(all_logits, axis=0)[:n]
        acc = batch_accuracy(self.y_test, logits_full)
        self.logger.info(f"Test accuracy: {acc:.4f}")
        return float(acc)


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

    board_host = general.get("board_host", "localhost")
    board_port = int(general.get("board_port", 50051))  # unified board default

    global_logger = setup_global_logger(run_dir, log_level)
    global_logger.info(f"Run dir: {run_dir}")
    global_logger.info(
        f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"m2_verbose={m2_verbose}, m1m3_verbose={m1m3_verbose}, bucket_board_addr={board_host}:{board_port} "
        f"model_architecture={model_arch}"
    )

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray_logging_level = logging.ERROR if suppress_warnings else logging.INFO
    ray_log_to_driver = not suppress_warnings
    ray.init(ignore_reinit_error=True, logging_level=ray_logging_level, log_to_driver=ray_log_to_driver)

    (x_train, y_train), (x_test, y_test) = load_mnist()

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

    x_shards = np.array_split(x_train, n_clients)
    y_shards = np.array_split(y_train, n_clients)

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
