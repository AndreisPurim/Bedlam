#!/usr/bin/env python3
"""
federated_client.py -- Toy federated learning (FedAvg) over the existing Board.

Flow:
 - A FedServer actor polls client updates from the Board (receiver=server_name).
 - Clients train locally and post model weights to the server.
 - Server aggregates weights (FedAvg) and sends a new global model to each client.

This reuses the Board as a simple message relay; the Board itself does not change.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import json
import logging
import time
from datetime import datetime

import numpy as np
import ray
import yaml
import tensorflow as tf
from keras import losses, optimizers

from client import load_dataset, setup_global_logger, setup_peer_logger
from models.factory import build_split_models
from peers.base import BaseBoardClient, stratified_split


def _weights_to_bytes(weights: list[np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, *weights)
    return buf.getvalue()


def _bytes_to_weights(blob: bytes) -> list[np.ndarray]:
    buf = io.BytesIO(blob)
    data = np.load(buf, allow_pickle=False)
    keys = sorted(data.files, key=lambda k: int(k.split("_")[1]))
    return [data[k] for k in keys]


def encode_fed_message(
    op: str,
    round_idx: int,
    sender: str,
    weights: list[np.ndarray],
    num_samples: int | None = None,
) -> bytes:
    header = {
        "op": op,
        "round": int(round_idx),
        "sender": sender,
    }
    if num_samples is not None:
        header["num_samples"] = int(num_samples)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_bytes).to_bytes(4, "big")
    return header_len + header_bytes + _weights_to_bytes(weights)


def decode_fed_message(blob: bytes) -> tuple[dict, list[np.ndarray]]:
    if len(blob) < 4:
        raise ValueError("payload too short for header length")
    hlen = int.from_bytes(blob[:4], "big")
    if len(blob) < 4 + hlen:
        raise ValueError("payload truncated before header")
    header = json.loads(blob[4:4 + hlen].decode("utf-8"))
    weights = _bytes_to_weights(blob[4 + hlen :])
    return header, weights


def build_full_model(
    model_arch: str,
    input_shape=(28, 28, 1),
    num_classes: int = 10,
):
    m1, m2, m3 = build_split_models(
        m1_name=model_arch,
        m2_name=model_arch,
        m3_name=model_arch,
        m1_kwargs={"input_shape": input_shape},
        m2_kwargs={"input_dim": 128},
        m3_kwargs={"input_dim": 64, "num_classes": num_classes},
    )
    inputs = tf.keras.Input(shape=input_shape)
    x = m1(inputs)
    x = m2(x)
    outputs = m3(x)
    return tf.keras.Model(inputs, outputs, name=f"full_{model_arch}")


@ray.remote
class FedServer:
    def __init__(
        self,
        name: str,
        run_dir: str,
        board_host: str,
        board_port: int,
        model_arch: str,
        input_shape: tuple[int, int, int],
        num_classes: int,
        client_names: list[str],
        rounds: int,
        min_updates: int,
        lr: float,
        log_level: str = "INFO",
    ):
        self.name = name
        self.rounds = int(rounds)
        self.min_updates = max(1, int(min_updates))
        self.client_names = list(client_names)
        self.logger = setup_peer_logger(name, run_dir, log_level)
        self.board = BaseBoardClient(board_host, board_port, logger=self.logger)
        self.model = build_full_model(
            model_arch=model_arch,
            input_shape=input_shape,
            num_classes=num_classes,
        )
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self._pending: dict[int, dict[str, tuple[int, list[np.ndarray]]]] = {}

    def _broadcast_global(self, round_idx: int, weights: list[np.ndarray]):
        payload = encode_fed_message("GLOBAL", round_idx, self.name, weights)
        for client in self.client_names:
            self.board.post_message(sender=self.name, receiver=client, payload=payload)
        self.logger.info("Broadcasted global weights for round %d to %d clients", round_idx, len(self.client_names))

    def _collect_updates(self, round_idx: int) -> list[tuple[int, list[np.ndarray]]]:
        updates: dict[str, tuple[int, list[np.ndarray]]] = {}
        if round_idx in self._pending:
            updates.update(self._pending.pop(round_idx))

        while len(updates) < self.min_updates:
            msg = self.board.poll_message(receiver=self.name)
            if not msg:
                time.sleep(0.01)
                continue
            try:
                header, weights = decode_fed_message(msg["payload"])
            except Exception:
                continue
            if header.get("op") != "UPDATE":
                continue
            r = int(header.get("round", -1))
            sender = str(header.get("sender", ""))
            num_samples = int(header.get("num_samples", 0))
            if r != round_idx:
                bucket = self._pending.setdefault(r, {})
                if sender and sender not in bucket:
                    bucket[sender] = (num_samples, weights)
                continue
            if sender in updates:
                continue
            updates[sender] = (num_samples, weights)
            self.logger.info("Received update round=%d sender=%s samples=%d", r, sender, num_samples)

        return list(updates.values())

    @staticmethod
    def _aggregate(updates: list[tuple[int, list[np.ndarray]]]) -> list[np.ndarray]:
        total = sum(max(1, num) for num, _ in updates)
        agg = [np.zeros_like(w) for w in updates[0][1]]
        for num_samples, weights in updates:
            weight = max(1, int(num_samples))
            for i, w in enumerate(weights):
                agg[i] += w * weight
        for i in range(len(agg)):
            agg[i] = agg[i] / float(total)
        return agg

    def run(self):
        # Round 0: broadcast initial weights
        self._broadcast_global(0, self.model.get_weights())

        for round_idx in range(self.rounds):
            updates = self._collect_updates(round_idx)
            new_weights = self._aggregate(updates)
            self.model.set_weights(new_weights)
            self._broadcast_global(round_idx + 1, new_weights)
        self.logger.info("FedServer finished %d rounds", self.rounds)
        return "fed_server_done"


@ray.remote
class FedClient:
    def __init__(
        self,
        name: str,
        run_dir: str,
        board_host: str,
        board_port: int,
        server_name: str,
        model_arch: str,
        input_shape: tuple[int, int, int],
        num_classes: int,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        rounds: int,
        local_epochs: int,
        batch_size: int,
        lr: float,
        log_level: str = "INFO",
    ):
        self.name = name
        self.server_name = server_name
        self.rounds = int(rounds)
        self.local_epochs = max(1, int(local_epochs))
        self.batch_size = int(batch_size)
        self.logger = setup_peer_logger(name, run_dir, log_level)
        self.board = BaseBoardClient(board_host, board_port, logger=self.logger)
        self.model = build_full_model(
            model_arch=model_arch,
            input_shape=input_shape,
            num_classes=num_classes,
        )
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self._pending: dict[int, list[np.ndarray]] = {}
        self.metrics_path = os.path.join(run_dir, f"metrics_fed_{name}.csv")
        with open(self.metrics_path, "w") as f:
            f.write("round,train_loss,train_acc,test_loss,test_acc\n")

    def _wait_for_global(self, round_idx: int) -> list[np.ndarray]:
        if round_idx in self._pending:
            return self._pending.pop(round_idx)
        while True:
            msg = self.board.poll_message(receiver=self.name)
            if not msg:
                time.sleep(0.01)
                continue
            try:
                header, weights = decode_fed_message(msg["payload"])
            except Exception:
                continue
            if header.get("op") != "GLOBAL":
                continue
            r = int(header.get("round", -1))
            if r != round_idx:
                self._pending[r] = weights
                continue
            return weights

    def train(self):
        # Initial global model (round 0)
        weights = self._wait_for_global(0)
        self.model.set_weights(weights)

        for round_idx in range(self.rounds):
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.local_epochs,
                batch_size=self.batch_size,
                verbose=0,
            )
            train_loss, train_acc = self.model.evaluate(
                self.x_train, self.y_train, batch_size=self.batch_size, verbose=0
            )
            test_loss, test_acc = self.model.evaluate(
                self.x_test, self.y_test, batch_size=self.batch_size, verbose=0
            )
            with open(self.metrics_path, "a") as f:
                f.write(f"{round_idx},{train_loss:.6f},{train_acc:.6f},{test_loss:.6f},{test_acc:.6f}\n")

            payload = encode_fed_message(
                "UPDATE",
                round_idx,
                self.name,
                self.model.get_weights(),
                num_samples=len(self.x_train),
            )
            self.board.post_message(sender=self.name, receiver=self.server_name, payload=payload)
            self.logger.info("Sent update round=%d samples=%d", round_idx, len(self.x_train))

            weights = self._wait_for_global(round_idx + 1)
            self.model.set_weights(weights)

        self.logger.info("FedClient finished %d rounds", self.rounds)
        return f"{self.name} done"


def main(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_cfg = cfg.get("run", {})
    general = cfg.get("general", {})
    peers_cfg = cfg.get("peers", {})
    fed_cfg = cfg.get("federated", {})

    base_dir = run_cfg.get("base_dir", "runs")
    run_name = run_cfg.get("name") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    rounds = int(fed_cfg.get("rounds", 5))
    local_epochs = int(fed_cfg.get("local_epochs", 1))
    server_name = fed_cfg.get("server_name", "fed_server")

    batch_size = int(general.get("batch_size", 128))
    lr = float(general.get("lr", 1e-3))
    suppress_warnings = bool(general.get("suppress_warnings", False))
    log_level = general.get("log_level", "INFO")
    model_arch = general.get("model_architecture", "default")
    dataset = general.get("dataset", "mnist")

    board_host = general.get("board_host", "localhost")
    board_port = int(general.get("board_port", 50051))

    global_logger = setup_global_logger(run_dir, log_level)
    global_logger.info(f"Run dir: {run_dir}")
    global_logger.info(
        f"Federated config: rounds={rounds}, local_epochs={local_epochs}, batch_size={batch_size}, lr={lr}, "
        f"model_architecture={model_arch}, dataset={dataset}, board={board_host}:{board_port}"
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
    if not m1m3_peers_cfg:
        raise ValueError("Federated mode requires peers.M1M3 entries for client names.")

    client_names = [p["name"] for p in m1m3_peers_cfg]
    min_updates = int(fed_cfg.get("min_updates", len(client_names)))

    x_shards, y_shards = stratified_split(x_train, y_train, len(client_names), seed=seed)
    input_shape = x_train.shape[1:]
    num_classes = int(len(np.unique(y_train)))

    server = FedServer.remote(
        name=server_name,
        run_dir=run_dir,
        board_host=board_host,
        board_port=board_port,
        model_arch=model_arch,
        input_shape=input_shape,
        num_classes=num_classes,
        client_names=client_names,
        rounds=rounds,
        min_updates=min_updates,
        lr=lr,
        log_level=log_level,
    )

    clients = []
    for idx, cfg_entry in enumerate(m1m3_peers_cfg):
        name = cfg_entry["name"]
        actor = FedClient.remote(
            name=name,
            run_dir=run_dir,
            board_host=board_host,
            board_port=board_port,
            server_name=server_name,
            model_arch=model_arch,
            input_shape=input_shape,
            num_classes=num_classes,
            x_train=x_shards[idx],
            y_train=y_shards[idx],
            x_test=x_test,
            y_test=y_test,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            log_level=log_level,
        )
        clients.append(actor)

    server_task = server.run.remote()
    ray.get([c.train.remote() for c in clients])
    ray.get(server_task)


if __name__ == "__main__":
    main()
