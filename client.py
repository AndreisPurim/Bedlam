#!/usr/bin/env python3
"""
client.py — Split Learning with Ray (M1/M3 clients + M2 server)
---------------------------------------------------------------

This script extends the working MNIST split model into a
multi-client split-learning setup using Ray:

  - One PeerM2 actor holds the middle model (M2).
  - Multiple PeerM1M3 actors each hold their own M1 + M3
    and their own local shard of the training data.

For each batch:
  Client:
    x -> M1 -> z_cut
    send z_cut to PeerM2.forward() -> z_mid
    z_mid -> M3 -> logits -> loss
    compute dL/dz_mid
    send dL/dz_mid to PeerM2.backward() -> dL/dz_cut
    backprop through M1 with dL/dz_cut

This gives you a real split-learning training loop with
two (or more) clients and one M2 "server" actor.

No gRPC, no OMR board yet — just Ray and TensorFlow.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import uuid
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, losses, optimizers
import ray


# ============================================================
# 1. Models: M1, M2, M3
# ============================================================

def build_M1(input_shape=(28, 28, 1)) -> keras.Model:
    """Early feature extractor (conceptually on the client)."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)   # z_cut
    return models.Model(inputs, x, name="M1")


def build_M2(input_dim=128) -> keras.Model:
    """Middle model (conceptually on the server)."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)  # z_mid
    return models.Model(inputs, x, name="M2")


def build_M3(input_dim=64, num_classes=10) -> keras.Model:
    """Classifier head (conceptually back on the client)."""
    inputs = layers.Input(shape=(input_dim,))
    outputs = layers.Dense(num_classes, activation="softmax")(inputs)
    return models.Model(inputs, outputs, name="M3")


# ============================================================
# 2. Data utilities
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
# 3. PeerM2: Ray actor holding M2 and its optimizer
# ============================================================

@ray.remote
class PeerM2:
    """
    Holds the middle model M2 and performs its part of
    forward and backward passes.

    Protocol:
      - forward(session_id, z_cut_np) -> z_mid_np
      - backward(session_id, dL_dz_mid_np) -> dL_dz_cut_np

    We store a gradient tape and the input z_cut per session_id
    so we can complete the backward when the client sends
    dL/dz_mid.
    """

    def __init__(self, input_dim=128, lr=1e-3):
        self.M2 = build_M2(input_dim=input_dim)
        self.opt_M2 = optimizers.Adam(learning_rate=lr)
        self._sessions = {}  # session_id -> (tape, z_cut, z_mid)

    def forward(self, session_id: str, z_cut_np: np.ndarray) -> np.ndarray:
        """Forward pass through M2, recording a tape for this session."""
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)

        # Record operations from z_cut -> z_mid
        with tf.GradientTape() as tape:
            tape.watch(z_cut)
            z_mid = self.M2(z_cut, training=True)

        # Cache for backward() call
        self._sessions[session_id] = (tape, z_cut, z_mid)

        return z_mid.numpy()

    def backward(self, session_id: str, dL_dz_mid_np: np.ndarray) -> np.ndarray:
        """
        Backward pass through M2 using upstream gradient dL/dz_mid.
        Returns dL/dz_cut for the client.
        """
        if session_id not in self._sessions:
            raise ValueError(f"No cached forward pass for session_id={session_id}")

        tape, z_cut, z_mid = self._sessions.pop(session_id)

        dL_dz_mid = tf.convert_to_tensor(dL_dz_mid_np, dtype=tf.float32)

        # Compute gradients w.r.t M2 parameters and z_cut
        targets = self.M2.trainable_variables + [z_cut]
        grads_all = tape.gradient(z_mid, targets, output_gradients=dL_dz_mid)

        grads_M2 = grads_all[:-1]
        dL_dz_cut = grads_all[-1]

        # Update M2 parameters
        self.opt_M2.apply_gradients(zip(grads_M2, self.M2.trainable_variables))

        return dL_dz_cut.numpy()

    def infer(self, z_cut_np: np.ndarray) -> np.ndarray:
        """Pure forward inference for evaluation: z_cut -> z_mid."""
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        z_mid = self.M2(z_cut, training=False)
        return z_mid.numpy()


# ============================================================
# 4. PeerM1M3: Ray actor holding M1 + M3
# ============================================================

@ray.remote
class PeerM1M3:
    """
    One client in the Split Learning system.

    Holds:
      - Local copy of M1 and M3.
      - Local shard of training data.
      - Test data (for final evaluation).
      - Reference to PeerM2 actor.
    """

    def __init__(
        self,
        client_id: int,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        server: ray.actor.ActorHandle,
        epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
    ):
        self.id = client_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.server = server
        self.epochs = epochs
        self.batch_size = batch_size

        # Local models and optimizers
        self.M1 = build_M1()
        self.M3 = build_M3(input_dim=64)
        self.opt_M1 = optimizers.Adam(learning_rate=lr)
        self.opt_M3 = optimizers.Adam(learning_rate=lr)
        self.loss_fn = losses.SparseCategoricalCrossentropy()

    # ----------------- training loop -----------------
    def train(self):
        n = len(self.x_train)
        steps_per_epoch = n // self.batch_size

        for epoch in range(1, self.epochs + 1):
            # shuffle local data
            idx = np.random.permutation(n)
            x_sh = self.x_train[idx]
            y_sh = self.y_train[idx]

            epoch_losses = []
            epoch_accs = []
            start = time.time()

            for step in range(steps_per_epoch):
                lo = step * self.batch_size
                hi = lo + self.batch_size
                xb = tf.convert_to_tensor(x_sh[lo:hi], dtype=tf.float32)
                yb = tf.convert_to_tensor(y_sh[lo:hi], dtype=tf.int32)

                # ----- forward on M1 -----
                with tf.GradientTape(persistent=True) as tape_M1:
                    z_cut = self.M1(xb, training=True)  # [B, 128]

                z_cut_np = z_cut.numpy()
                session_id = f"{self.id}-{epoch}-{step}-{uuid.uuid4().hex}"

                # ----- send z_cut to server (M2 forward) -----
                z_mid_np = ray.get(self.server.forward.remote(session_id, z_cut_np))
                z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)

                # ----- forward on M3 + loss, grad wrt z_mid -----
                with tf.GradientTape() as tape_M3:
                    tape_M3.watch(z_mid)
                    logits = self.M3(z_mid, training=True)  # [B, 10]
                    loss_value = self.loss_fn(yb, logits)

                targets = self.M3.trainable_variables + [z_mid]
                grads_all = tape_M3.gradient(loss_value, targets)
                grads_M3 = grads_all[:-1]
                dL_dz_mid = grads_all[-1]

                # update local M3
                self.opt_M3.apply_gradients(zip(grads_M3, self.M3.trainable_variables))

                # ----- send dL/dz_mid to server (M2 backward) -----
                dL_dz_mid_np = dL_dz_mid.numpy()
                dL_dz_cut_np = ray.get(self.server.backward.remote(session_id, dL_dz_mid_np))

                dL_dz_cut = tf.convert_to_tensor(dL_dz_cut_np, dtype=tf.float32)

                # ----- backprop through M1 -----
                grads_M1 = tape_M1.gradient(
                    z_cut, self.M1.trainable_variables, output_gradients=dL_dz_cut
                )
                self.opt_M1.apply_gradients(zip(grads_M1, self.M1.trainable_variables))
                del tape_M1  # free resources

                # ----- metrics -----
                acc_batch = batch_accuracy(yb.numpy(), logits.numpy())
                epoch_losses.append(float(loss_value.numpy()))
                epoch_accs.append(acc_batch)

                if step % 100 == 0:
                    print(
                        f"[Client {self.id}] Epoch {epoch} "
                        f"Step {step}/{steps_per_epoch} "
                        f"Loss={epoch_losses[-1]:.4f} Acc={acc_batch:.4f}"
                    )

            mean_loss = float(np.mean(epoch_losses))
            mean_acc = float(np.mean(epoch_accs))
            elapsed = time.time() - start
            print(
                f"[Client {self.id}] Epoch {epoch} done in {elapsed:.1f}s "
                f"→ Loss={mean_loss:.4f} Acc={mean_acc:.4f}"
            )

        return f"Client {self.id} training finished."

    # ----------------- evaluation -----------------
    def evaluate(self) -> float:
        """Run full evaluation using M1 -> PeerM2 -> M3."""
        batch_size = self.batch_size
        n = len(self.x_test)
        all_logits = []

        for i in range(0, n, batch_size):
            xb = tf.convert_to_tensor(self.x_test[i:i + batch_size], dtype=tf.float32)
            z_cut = self.M1(xb, training=False)
            z_cut_np = z_cut.numpy()

            # use server in inference mode
            z_mid_np = ray.get(self.server.infer.remote(z_cut_np))
            z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)
            logits = self.M3(z_mid, training=False)
            all_logits.append(logits.numpy())

        logits_full = np.concatenate(all_logits, axis=0)[:n]
        acc = batch_accuracy(self.y_test, logits_full)
        print(f"[Client {self.id}] Test accuracy: {acc:.4f}")
        return float(acc)


# ============================================================
# 5. Main: spin up Ray, server, and multiple clients
# ============================================================

def main():
    ray.init(ignore_reinit_error=True)

    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Number of clients with their own M1/M3 and local training shard
    n_clients = 2
    epochs = 2
    batch_size = 128
    lr = 1e-3

    # Split training data into n_clients shards
    x_shards = np.array_split(x_train, n_clients)
    y_shards = np.array_split(y_train, n_clients)

    # Shared server M2
    server = PeerM2.remote(input_dim=128, lr=lr)

    # Create clients
    clients = []
    for i in range(n_clients):
        c = PeerM1M3.remote(
            client_id=i + 1,
            x_train=x_shards[i],
            y_train=y_shards[i],
            x_test=x_test,
            y_test=y_test,
            server=server,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        clients.append(c)

    # Train all clients (in parallel)
    print("Starting training for all clients...")
    _ = ray.get([c.train.remote() for c in clients])

    # Evaluate each client separately
    print("\nEvaluating clients on test set...")
    accs = ray.get([c.evaluate.remote() for c in clients])
    for i, acc in enumerate(accs, start=1):
        print(f"Client {i} final test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
