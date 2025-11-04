#!/usr/bin/env python3
"""
client.py — Split Learning over Ray with Board + Config + Run Folders
---------------------------------------------------------------------

Topology:

  [PeerM1M3]  <-->  [Board]  <-->  [PeerM2]

- Multiple PeerM1M3 actors (each with its own M1 + M3 + data shard)
- Multiple PeerM2 actors (each with its own M2)
- All communication is via the Board actor using message objects:
    kind: "fwd_req" | "fwd_res" | "bwd_req" | "bwd_res" | "infer_req" | "infer_res"
    msg_id: unique UUID
    session_id: training step identifier
    sender / receiver: peer names
    payload: numpy array
    timestamp: float

Logging:
- Each run has its own folder: runs/run_YYYYmmdd_HHMMSS/
- Global log: global.log
- Board log: board.log
- Per-peer logs: <peer_name>.log

Configured via config.yaml.

Author: you & GPT
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # suppress TF INFO
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # force CPU use

import time
import uuid
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, losses, optimizers
import ray
import yaml


# ============================================================
# 1. Logging helpers
# ============================================================

def setup_global_logger(run_dir: str, level: str = "INFO") -> logging.Logger:
    """Root/global logger writing to stdout + run_dir/global.log."""
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger()  # root logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Global file
    fh = logging.FileHandler(os.path.join(run_dir, "global.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def setup_peer_logger(peer_name: str, run_dir: str, level: str = "INFO") -> logging.Logger:
    """
    Logger dedicated to a peer (or the board). Logs only to its own file:
      runs/run_xxxx/<peer_name>.log
    """
    logger = logging.getLogger(peer_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False  # don’t bubble to root (avoid duplicates)

    fmt = logging.Formatter(f"[{peer_name}] %(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(os.path.join(run_dir, f"{peer_name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# 2. Models: M1, M2, M3
# ============================================================

def build_M1(input_shape=(28, 28, 1)) -> keras.Model:
    """Early feature extractor (client-side)."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)   # z_cut
    return models.Model(inputs, x, name="M1")


def build_M2(input_dim=128) -> keras.Model:
    """Middle model (peer holding the 'server' part)."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)  # z_mid
    return models.Model(inputs, x, name="M2")


def build_M3(input_dim=64, num_classes=10) -> keras.Model:
    """Classifier head (back on the client)."""
    inputs = layers.Input(shape=(input_dim,))
    outputs = layers.Dense(num_classes, activation="softmax")(inputs)
    return models.Model(inputs, outputs, name="M3")


# ============================================================
# 3. Data utilities
# ============================================================

def load_mnist(normalize=True):
    """Loads MNIST and adds channel dimension."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    if normalize:
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def batch_accuracy(y_true, y_pred):
    """Simple NumPy accuracy for a batch."""
    return np.mean(np.argmax(y_pred, axis=1) == y_true)


# ============================================================
# 4. Board actor
# ============================================================

@ray.remote
class Board:
    """
    Central message board.

    Messages are stored in queues keyed by:
      - kind (fwd_req, fwd_res, bwd_req, bwd_res, infer_req, infer_res)
      - receiver (peer name)

    API:
      post_message(kind, sender, receiver, session_id, payload) -> msg_id
      poll_message(kind, receiver, session_id=None) -> message | None
    """

    def __init__(
        self,
        name: str,
        run_dir: str,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 100,
    ):
        self.name = name
        self.logger = setup_peer_logger(name, run_dir, log_level)
        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self._post_count = 0
        self._poll_count = 0

        # queues[kind][receiver] = [message, ...]
        self.queues = defaultdict(lambda: defaultdict(list))

        self.logger.info(
            f"Board initialized with verbose={self.verbose}, log_every={self.log_every}"
        )

    def post_message(self, kind: str, sender: str, receiver: str, session_id: str, payload: np.ndarray) -> str:
        """Store a message and return its msg_id."""
        self._post_count += 1
        msg_id = uuid.uuid4().hex
        message = {
            "msg_id": msg_id,
            "kind": kind,
            "sender": sender,
            "receiver": receiver,
            "session_id": session_id,
            "payload": payload,
            "timestamp": time.time(),
        }
        self.queues[kind][receiver].append(message)

        if self.verbose and (self._post_count % self.log_every == 0):
            shape = getattr(payload, "shape", None)
            self.logger.info(
                f"[POST #{self._post_count}] kind={kind} sender={sender} "
                f"receiver={receiver} session={session_id} shape={shape} msg_id={msg_id}"
            )

        return msg_id

    def poll_message(self, kind: str, receiver: str, session_id: str | None = None):
        """
        Non-blocking poll for a single message of given kind + receiver.

        If session_id is given, returns the earliest message for that session.
        Otherwise, returns the earliest message for that receiver.
        """
        self._poll_count += 1

        kind_queues = self.queues.get(kind)
        if not kind_queues:
            return None

        msgs = kind_queues.get(receiver)
        if not msgs:
            return None

        # Find matching message
        if session_id is None:
            msg = msgs.pop(0)
        else:
            idx = None
            for i, m in enumerate(msgs):
                if m["session_id"] == session_id:
                    idx = i
                    break
            if idx is None:
                return None
            msg = msgs.pop(idx)

        if self.verbose and (self._poll_count % self.log_every == 0):
            shape = getattr(msg["payload"], "shape", None)
            self.logger.info(
                f"[POLL #{self._poll_count}] kind={kind} receiver={receiver} "
                f"session={msg['session_id']} msg_id={msg['msg_id']} shape={shape}"
            )

        return msg


# ============================================================
# 5. Ray peer: M2 holder
# ============================================================

@ray.remote
class PeerM2:
    """
    Peer that holds M2 and performs its part of the computation.

    It does NOT get called directly by clients.
    Instead, it runs a loop where it:

      - polls the Board for fwd_req (from any M1M3)
      - computes z_mid, caches tape and z_cut
      - posts fwd_res back to the Board
      - polls the Board for bwd_req
      - computes grads, updates M2, posts bwd_res back

    It also handles infer_req / infer_res for evaluation.
    """

    def __init__(
        self,
        name: str,
        run_dir: str,
        board,
        input_dim: int = 128,
        lr: float = 1e-3,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
    ):
        self.name = name
        self.logger = setup_peer_logger(name, run_dir, log_level)
        self.board = board

        self.M2 = build_M2(input_dim=input_dim)
        self.opt_M2 = optimizers.Adam(learning_rate=lr)
        self._sessions = {}  # session_id -> (tape, z_cut, z_mid)

        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0

        self.logger.info(
            f"Initialized PeerM2 with input_dim={input_dim}, lr={lr}, "
            f"verbose={self.verbose}, log_every={self.log_every}"
        )

    # ---------------- main processing loop ----------------

    def run(self):
        """
        Main loop: process fwd_req, bwd_req, infer_req from the Board.
        This is intended to run "forever" (until Ray shuts down).
        """
        self.logger.info("PeerM2.run() loop started.")
        while True:
            handled = False

            # 1) Forward requests
            fwd_msg = ray.get(self.board.poll_message.remote(
                "fwd_req", receiver=self.name, session_id=None
            ))
            if fwd_msg is not None:
                self._handle_forward(fwd_msg)
                handled = True

            # 2) Backward requests
            bwd_msg = ray.get(self.board.poll_message.remote(
                "bwd_req", receiver=self.name, session_id=None
            ))
            if bwd_msg is not None:
                self._handle_backward(bwd_msg)
                handled = True

            # 3) Inference requests
            infer_msg = ray.get(self.board.poll_message.remote(
                "infer_req", receiver=self.name, session_id=None
            ))
            if infer_msg is not None:
                self._handle_infer(infer_msg)
                handled = True

            if not handled:
                time.sleep(0.01)  # idle wait

    # ---------------- internal handlers ----------------

    def _handle_forward(self, msg: dict):
        self._fwd_count += 1
        session_id = msg["session_id"]
        z_cut_np = msg["payload"]
        sender = msg["sender"]

        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z_cut)
            z_mid = self.M2(z_cut, training=True)

        self._sessions[session_id] = (tape, z_cut, z_mid)
        z_mid_np = z_mid.numpy()

        # send response
        ray.get(self.board.post_message.remote(
            "fwd_res",
            sender=self.name,
            receiver=sender,
            session_id=session_id,
            payload=z_mid_np,
        ))

        if self.verbose and (self._fwd_count % self.log_every == 0):
            bs = z_cut_np.shape[0]
            feat = z_cut_np.shape[1] if z_cut_np.ndim > 1 else 1
            self.logger.info(
                f"[FWD #{self._fwd_count}] session={session_id} from={sender} "
                f"batch={bs} feat_dim={feat}"
            )

    def _handle_backward(self, msg: dict):
        self._bwd_count += 1
        session_id = msg["session_id"]
        dL_dz_mid_np = msg["payload"]
        sender = msg["sender"]

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

        ray.get(self.board.post_message.remote(
            "bwd_res",
            sender=self.name,
            receiver=sender,
            session_id=session_id,
            payload=dL_dz_cut_np,
        ))

        if self.verbose and (self._bwd_count % self.log_every == 0):
            grad_norm = tf.linalg.global_norm(grads_M2).numpy()
            dzcut_norm = tf.linalg.global_norm([dL_dz_cut]).numpy()
            self.logger.info(
                f"[BWD #{self._bwd_count}] session={session_id} from={sender} "
                f"grad_norm(M2)={grad_norm:.4f} grad_norm(dL/dz_cut)={dzcut_norm:.4f}"
            )

    def _handle_infer(self, msg: dict):
        self._infer_count += 1
        session_id = msg["session_id"]
        z_cut_np = msg["payload"]
        sender = msg["sender"]

        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        z_mid = self.M2(z_cut, training=False)
        z_mid_np = z_mid.numpy()

        ray.get(self.board.post_message.remote(
            "infer_res",
            sender=self.name,
            receiver=sender,
            session_id=session_id,
            payload=z_mid_np,
        ))

        if self.verbose and (self._infer_count % self.log_every == 0):
            bs = z_cut_np.shape[0]
            self.logger.info(
                f"[INFER #{self._infer_count}] session={session_id} from={sender} batch={bs}"
            )


# ============================================================
# 6. Ray peer: M1+M3 client
# ============================================================

@ray.remote
class PeerM1M3:
    """
    A split-learning client peer (M1 + M3).

    - Holds M1 and M3.
    - Holds its own local training shard.
    - Knows:
        - the Board actor
        - the target M2 peer name (string)
    """

    def __init__(
        self,
        name: str,
        run_dir: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        board,
        target_m2: str,
        epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
    ):
        self.name = name
        self.logger = setup_peer_logger(name, run_dir, log_level)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.board = board
        self.target_m2 = target_m2

        self.epochs = epochs
        self.batch_size = batch_size

        self.M1 = build_M1()
        self.M3 = build_M3(input_dim=64)
        self.opt_M1 = optimizers.Adam(learning_rate=lr)
        self.opt_M3 = optimizers.Adam(learning_rate=lr)
        self.loss_fn = losses.SparseCategoricalCrossentropy()

        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0

        self.logger.info(
            f"Initialized PeerM1M3 epochs={epochs} batch_size={batch_size} lr={lr} "
            f"target_m2={target_m2} verbose={self.verbose} log_every={self.log_every}"
        )

    # ---------------- training ----------------

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

                # ----- M1 forward -----
                with tf.GradientTape(persistent=True) as tape_M1:
                    z_cut = self.M1(xb, training=True)
                z_cut_np = z_cut.numpy()

                session_id = f"{self.name}-{epoch}-{step}-{uuid.uuid4().hex}"

                # ----- send forward request to Board -----
                self._fwd_count += 1
                ray.get(self.board.post_message.remote(
                    "fwd_req",
                    sender=self.name,
                    receiver=self.target_m2,
                    session_id=session_id,
                    payload=z_cut_np,
                ))
                if self.verbose and (self._fwd_count % self.log_every == 0):
                    self.logger.info(
                        f"[FWD_REQ #{self._fwd_count}] session={session_id} "
                        f"to={self.target_m2} batch={z_cut_np.shape[0]}"
                    )

                # ----- wait for forward response (z_mid) -----
                z_mid_np = None
                while z_mid_np is None:
                    msg = ray.get(self.board.poll_message.remote(
                        "fwd_res",
                        receiver=self.name,
                        session_id=session_id,
                    ))
                    if msg is not None:
                        z_mid_np = msg["payload"]
                        if self.verbose and (self._fwd_count % self.log_every == 0):
                            self.logger.info(
                                f"[FWD_RES #{self._fwd_count}] session={session_id} "
                                f"from={msg['sender']} shape={z_mid_np.shape}"
                            )
                        break
                    time.sleep(0.01)

                z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)

                # ----- M3 forward + loss, grad wrt z_mid -----
                with tf.GradientTape() as tape_M3:
                    tape_M3.watch(z_mid)
                    logits = self.M3(z_mid, training=True)
                    loss_value = self.loss_fn(yb, logits)

                targets = self.M3.trainable_variables + [z_mid]
                grads_all = tape_M3.gradient(loss_value, targets)
                grads_M3 = grads_all[:-1]
                dL_dz_mid = grads_all[-1]
                self.opt_M3.apply_gradients(zip(grads_M3, self.M3.trainable_variables))

                # ----- send backward request (dL/dz_mid) -----
                self._bwd_count += 1
                dL_dz_mid_np = dL_dz_mid.numpy()
                ray.get(self.board.post_message.remote(
                    "bwd_req",
                    sender=self.name,
                    receiver=self.target_m2,
                    session_id=session_id,
                    payload=dL_dz_mid_np,
                ))
                if self.verbose and (self._bwd_count % self.log_every == 0):
                    self.logger.info(
                        f"[BWD_REQ #{self._bwd_count}] session={session_id} "
                        f"to={self.target_m2}"
                    )

                # ----- wait for backward response (dL/dz_cut) -----
                dL_dz_cut_np = None
                while dL_dz_cut_np is None:
                    msg = ray.get(self.board.poll_message.remote(
                        "bwd_res",
                        receiver=self.name,
                        session_id=session_id,
                    ))
                    if msg is not None:
                        dL_dz_cut_np = msg["payload"]
                        if self.verbose and (self._bwd_count % self.log_every == 0):
                            self.logger.info(
                                f"[BWD_RES #{self._bwd_count}] session={session_id} "
                                f"from={msg['sender']}"
                            )
                        break
                    time.sleep(0.01)

                dL_dz_cut = tf.convert_to_tensor(dL_dz_cut_np, dtype=tf.float32)

                # ----- backprop through M1 -----
                grads_M1 = tape_M1.gradient(
                    z_cut, self.M1.trainable_variables, output_gradients=dL_dz_cut
                )
                self.opt_M1.apply_gradients(zip(grads_M1, self.M1.trainable_variables))
                del tape_M1

                # ----- metrics -----
                acc_batch = batch_accuracy(yb.numpy(), logits.numpy())
                epoch_losses.append(float(loss_value.numpy()))
                epoch_accs.append(acc_batch)

                if step % 100 == 0:
                    self.logger.info(
                        f"Epoch {epoch} Step {step}/{steps_per_epoch} "
                        f"Loss={epoch_losses[-1]:.4f} Acc={acc_batch:.4f}"
                    )

            mean_loss = float(np.mean(epoch_losses))
            mean_acc = float(np.mean(epoch_accs))
            elapsed = time.time() - start
            self.logger.info(
                f"Epoch {epoch} done in {elapsed:.1f}s "
                f"→ Loss={mean_loss:.4f} Acc={mean_acc:.4f}"
            )

        return f"{self.name} training finished."

    # ---------------- evaluation ----------------

    def evaluate(self) -> float:
        """Full evaluation: x -> M1 -> Board -> M2 -> Board -> M3."""
        batch_size = self.batch_size
        n = len(self.x_test)
        all_logits = []

        for i in range(0, n, batch_size):
            xb = tf.convert_to_tensor(self.x_test[i:i + batch_size], dtype=tf.float32)
            z_cut = self.M1(xb, training=False)
            z_cut_np = z_cut.numpy()

            session_id = f"{self.name}-eval-{i}-{uuid.uuid4().hex}"

            # send infer request
            self._infer_count += 1
            ray.get(self.board.post_message.remote(
                "infer_req",
                sender=self.name,
                receiver=self.target_m2,
                session_id=session_id,
                payload=z_cut_np,
            ))
            if self.verbose and (self._infer_count % self.log_every == 0):
                self.logger.info(
                    f"[INFER_REQ #{self._infer_count}] session={session_id} "
                    f"to={self.target_m2} batch={z_cut_np.shape[0]}"
                )

            # wait for infer response
            z_mid_np = None
            while z_mid_np is None:
                msg = ray.get(self.board.poll_message.remote(
                    "infer_res",
                    receiver=self.name,
                    session_id=session_id,
                ))
                if msg is not None:
                    z_mid_np = msg["payload"]
                    if self.verbose and (self._infer_count % self.log_every == 0):
                        self.logger.info(
                            f"[INFER_RES #{self._infer_count}] session={session_id} "
                            f"from={msg['sender']} shape={z_mid_np.shape}"
                        )
                    break
                time.sleep(0.01)

            z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)
            logits = self.M3(z_mid, training=False)
            all_logits.append(logits.numpy())

        logits_full = np.concatenate(all_logits, axis=0)[:n]
        acc = batch_accuracy(self.y_test, logits_full)
        self.logger.info(f"Test accuracy: {acc:.4f}")
        return float(acc)


# ============================================================
# 7. Main with config + run folder + suppression flag
# ============================================================

def main(config_path: str = "config.yaml"):
    # ---- Load config ----
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_cfg = cfg.get("run", {})
    general = cfg.get("general", {})
    peers_cfg = cfg.get("peers", {})

    # ---- Run directory ----
    base_dir = run_cfg.get("base_dir", "runs")
    run_name = run_cfg.get("name")
    if not run_name:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # ---- Params ----
    epochs = int(general.get("epochs", 2))
    batch_size = int(general.get("batch_size", 128))
    lr = float(general.get("lr", 1e-3))
    suppress_warnings = bool(general.get("suppress_warnings", False))
    log_level = general.get("log_level", "INFO")

    m2_verbose = bool(general.get("m2_verbose", False))
    m2_log_every = int(general.get("m2_log_every", 50))

    m1m3_verbose = bool(general.get("m1m3_verbose", False))
    m1m3_log_every = int(general.get("m1m3_log_every", 50))

    board_verbose = bool(general.get("board_verbose", False))
    board_log_every = int(general.get("board_log_every", 100))

    # ---- Global logger ----
    global_logger = setup_global_logger(run_dir, log_level)
    global_logger.info(f"Run dir: {run_dir}")
    global_logger.info(
        f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"m2_verbose={m2_verbose}, m1m3_verbose={m1m3_verbose}, board_verbose={board_verbose}"
    )

    # ---- Control Ray warnings ----
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray_logging_level = logging.ERROR if suppress_warnings else logging.INFO
    ray_log_to_driver = not suppress_warnings

    ray.init(
        ignore_reinit_error=True,
        logging_level=ray_logging_level,
        log_to_driver=ray_log_to_driver,
    )

    (x_train, y_train), (x_test, y_test) = load_mnist()

    m1m3_peers_cfg = peers_cfg.get("M1M3", [])
    m2_peers_cfg = peers_cfg.get("M2", [])

    if not m1m3_peers_cfg or not m2_peers_cfg:
        raise ValueError("Config must define at least one M1M3 peer and one M2 peer.")

    n_clients = len(m1m3_peers_cfg)
    global_logger.info(f"Configured {n_clients} M1M3 peers and {len(m2_peers_cfg)} M2 peers.")

    # ---- Board actor ----
    board = Board.remote(
        name="board",
        run_dir=run_dir,
        log_level=log_level,
        verbose=board_verbose,
        log_every=board_log_every,
    )
    global_logger.info("Spawned Board actor.")

    # Shard training data across M1M3 peers
    x_shards = np.array_split(x_train, n_clients)
    y_shards = np.array_split(y_train, n_clients)

    # Create M2 peers
    m2_peers = {}
    for m2_cfg in m2_peers_cfg:
        name = m2_cfg["name"]
        m2_peer = PeerM2.remote(
            name=name,
            run_dir=run_dir,
            board=board,
            input_dim=128,
            lr=lr,
            log_level=log_level,
            verbose=m2_verbose,
            log_every=m2_log_every,
        )
        m2_peers[name] = m2_peer
        global_logger.info(f"Spawned M2 peer: {name}")

    # Launch M2 processing loops (fire-and-forget)
    for name, m2_peer in m2_peers.items():
        m2_peer.run.remote()
        global_logger.info(f"Started run() loop for M2 peer: {name}")

    # Create M1M3 peers, binding each to its configured M2 peer name
    clients = []
    for i, c_cfg in enumerate(m1m3_peers_cfg):
        name = c_cfg["name"]
        target_m2 = c_cfg["target_m2"]
        if target_m2 not in m2_peers:
            raise ValueError(f"M1M3 peer {name} references unknown M2 peer '{target_m2}'")

        client = PeerM1M3.remote(
            name=name,
            run_dir=run_dir,
            x_train=x_shards[i],
            y_train=y_shards[i],
            x_test=x_test,
            y_test=y_test,
            board=board,
            target_m2=target_m2,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            log_level=log_level,
            verbose=m1m3_verbose,
            log_every=m1m3_log_every,
        )
        clients.append(client)
        global_logger.info(f"Spawned M1M3 peer: {name} → bound to M2 peer: {target_m2}")

    # Train all clients
    global_logger.info("Starting training for all clients...")
    ray.get([c.train.remote() for c in clients])

    # Evaluate each client
    global_logger.info("Evaluating clients on test set...")
    accs = ray.get([c.evaluate.remote() for c in clients])
    for c_cfg, acc in zip(m1m3_peers_cfg, accs):
        global_logger.info(f"Client {c_cfg['name']} final test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
