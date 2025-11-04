#!/usr/bin/env python3
"""
client.py — Split Learning over Ray with Config + Run Folders
-------------------------------------------------------------

- Multiple PeerM1M3 actors (each with its own M1 + M3 + data shard)
- Multiple PeerM2 actors (each with its own M2)
- Each M1M3 peer is bound (via config) to one M2 peer.
- All communication via Ray (no gRPC / board yet).

Logging:
- Each run has its own folder: runs/run_YYYYmmdd_HHMMSS/
- Global log: global.log
- Per-peer logs: <peer_name>.log

Configured via config.yaml.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # suppress TF INFO
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # force CPU use

import time
import uuid
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, losses, optimizers
import ray
import yaml
from datetime import datetime


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
    Logger dedicated to a peer. Logs only to its own file:
      runs/run_xxxx/<peer_name>.log
    """
    logger = logging.getLogger(peer_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False  # don’t bubble up to root to avoid duplicates

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
# 4. Ray peer: M2 holder
# ============================================================

@ray.remote
class PeerM2:
    """
    Peer that holds M2 and performs its part of the computation.

    Protocol:
      - forward(session_id, z_cut_np) -> z_mid_np
      - backward(session_id, dL_dz_mid_np) -> dL_dz_cut_np

    We cache a tape per session so we can backprop M2 and dL/dz_cut
    when the client sends the upstream gradient.

    With verbosity enabled, logs:
      - when a forward is received
      - when a backward is processed (with some stats)
    """

    def __init__(
        self,
        name: str,
        run_dir: str,
        input_dim: int = 128,
        lr: float = 1e-3,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
    ):
        self.name = name
        self.logger = setup_peer_logger(name, run_dir, log_level)
        self.M2 = build_M2(input_dim=input_dim)
        self.opt_M2 = optimizers.Adam(learning_rate=lr)
        self._sessions = {}  # session_id -> (tape, z_cut, z_mid)

        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self._fwd_count = 0
        self._bwd_count = 0

        self.logger.info(
            f"Initialized PeerM2 with input_dim={input_dim}, lr={lr}, "
            f"verbose={self.verbose}, log_every={self.log_every}"
        )

    def forward(self, session_id: str, z_cut_np: np.ndarray) -> np.ndarray:
        """Forward pass through M2, recording a tape for this session."""
        self._fwd_count += 1

        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z_cut)
            z_mid = self.M2(z_cut, training=True)

        self._sessions[session_id] = (tape, z_cut, z_mid)

        if self.verbose and (self._fwd_count % self.log_every == 0):
            batch_size = z_cut_np.shape[0]
            feat_dim = z_cut_np.shape[1] if z_cut_np.ndim > 1 else 1
            self.logger.info(
                f"[FWD #{self._fwd_count}] session={session_id} "
                f"batch={batch_size} feat_dim={feat_dim}"
            )

        return z_mid.numpy()

    def backward(self, session_id: str, dL_dz_mid_np: np.ndarray) -> np.ndarray:
        """
        Backward pass through M2 using upstream gradient dL/dz_mid.
        Returns dL/dz_cut for the client.
        """
        self._bwd_count += 1

        if session_id not in self._sessions:
            raise ValueError(f"No cached forward for session_id={session_id}")
        tape, z_cut, z_mid = self._sessions.pop(session_id)

        dL_dz_mid = tf.convert_to_tensor(dL_dz_mid_np, dtype=tf.float32)
        targets = self.M2.trainable_variables + [z_cut]
        grads_all = tape.gradient(z_mid, targets, output_gradients=dL_dz_mid)
        grads_M2 = grads_all[:-1]
        dL_dz_cut = grads_all[-1]

        self.opt_M2.apply_gradients(zip(grads_M2, self.M2.trainable_variables))

        if self.verbose and (self._bwd_count % self.log_every == 0):
            # A couple of simple scalar stats for debugging
            grad_norm = tf.linalg.global_norm(grads_M2).numpy()
            dzcut_norm = tf.linalg.global_norm([dL_dz_cut]).numpy()
            self.logger.info(
                f"[BWD #{self._bwd_count}] session={session_id} "
                f"grad_norm(M2)={grad_norm:.4f} grad_norm(dL/dz_cut)={dzcut_norm:.4f}"
            )

        return dL_dz_cut.numpy()

    def infer(self, z_cut_np: np.ndarray) -> np.ndarray:
        """Forward-only pass for evaluation."""
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        z_mid = self.M2(z_cut, training=False)
        return z_mid.numpy()



# ============================================================
# 5. Ray peer: M1+M3 client
# ============================================================

@ray.remote
class PeerM1M3:
    """
    A split-learning client peer:

    - Holds M1 and M3.
    - Holds its own local training shard.
    - Knows the PeerM2 actor it is bound to (by config).
    """

    def __init__(
        self,
        name: str,
        run_dir: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        m2_peer,
        epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        log_level: str = "INFO",
    ):
        self.name = name
        self.logger = setup_peer_logger(name, run_dir, log_level)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.m2_peer = m2_peer
        self.epochs = epochs
        self.batch_size = batch_size

        self.M1 = build_M1()
        self.M3 = build_M3(input_dim=64)
        self.opt_M1 = optimizers.Adam(learning_rate=lr)
        self.opt_M3 = optimizers.Adam(learning_rate=lr)
        self.loss_fn = losses.SparseCategoricalCrossentropy()

        self.logger.info(
            f"Initialized PeerM1M3 epochs={epochs} batch_size={batch_size} lr={lr} "
            f"bound_to_m2_peer={m2_peer}"
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

                # ----- M1 forward -----
                with tf.GradientTape(persistent=True) as tape_M1:
                    z_cut = self.M1(xb, training=True)
                z_cut_np = z_cut.numpy()

                session_id = f"{self.name}-{epoch}-{step}-{uuid.uuid4().hex}"

                # ----- call M2 peer: forward -----
                z_mid_np = ray.get(self.m2_peer.forward.remote(session_id, z_cut_np))
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

                # ----- call M2 peer: backward -----
                dL_dz_mid_np = dL_dz_mid.numpy()
                dL_dz_cut_np = ray.get(self.m2_peer.backward.remote(session_id, dL_dz_mid_np))
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

    def evaluate(self) -> float:
        """Full evaluation: x -> M1 -> PeerM2 -> M3."""
        batch_size = self.batch_size
        n = len(self.x_test)
        all_logits = []

        for i in range(0, n, batch_size):
            xb = tf.convert_to_tensor(self.x_test[i:i + batch_size], dtype=tf.float32)
            z_cut = self.M1(xb, training=False)
            z_cut_np = z_cut.numpy()

            z_mid_np = ray.get(self.m2_peer.infer.remote(z_cut_np))
            z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)
            logits = self.M3(z_mid, training=False)
            all_logits.append(logits.numpy())

        logits_full = np.concatenate(all_logits, axis=0)[:n]
        acc = batch_accuracy(self.y_test, logits_full)
        self.logger.info(f"Test accuracy: {acc:.4f}")
        return float(acc)


# ============================================================
# 6. Main with config + run folder + suppression flag
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

    # ---- Global logger ----
    global_logger = setup_global_logger(run_dir, log_level)
    global_logger.info(f"Run dir: {run_dir}")
    global_logger.info(f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}")

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

    # NEW: read M2 verbosity settings
    m2_verbose = bool(general.get("m2_verbose", False))
    m2_log_every = int(general.get("m2_log_every", 50))

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
            input_dim=128,
            lr=lr,
            log_level=log_level,
            verbose=m2_verbose,       # <-- NEW
            log_every=m2_log_every,   # <-- NEW
        )
        m2_peers[name] = m2_peer
        global_logger.info(f"Spawned M2 peer: {name}")


    # Create M1M3 peers, binding each to its configured M2 peer
    clients = []
    for i, c_cfg in enumerate(m1m3_peers_cfg):
        name = c_cfg["name"]
        target_m2 = c_cfg["target_m2"]
        if target_m2 not in m2_peers:
            raise ValueError(f"M1M3 peer {name} references unknown M2 peer '{target_m2}'")

        m2_peer_handle = m2_peers[target_m2]
        client = PeerM1M3.remote(
            name=name,
            run_dir=run_dir,
            x_train=x_shards[i],
            y_train=y_shards[i],
            x_test=x_test,
            y_test=y_test,
            m2_peer=m2_peer_handle,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            log_level=log_level,
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
