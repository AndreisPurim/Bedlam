#!/usr/bin/env python3
"""
client.py — Minimal Split Learning Baseline (Fixed)
---------------------------------------------------
Validates that a three-part split neural network
(M1 → M2 → M3) can train end-to-end on MNIST.

No Ray, no gRPC — just local training with a clean design.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, losses, optimizers


# ============================================================
# 1. Model definitions
# ============================================================

def build_M1(input_shape=(28, 28, 1)) -> keras.Model:
    """Early feature extractor (conceptually: client-side)."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)   # z_cut
    return models.Model(inputs, x, name="M1")


def build_M2(input_dim=128) -> keras.Model:
    """Intermediate model (conceptually: server-side)."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)  # z_mid
    return models.Model(inputs, x, name="M2")


def build_M3(input_dim=64, num_classes=10) -> keras.Model:
    """Classifier head (conceptually: back on client)."""
    inputs = layers.Input(shape=(input_dim,))
    outputs = layers.Dense(num_classes, activation="softmax")(inputs)
    return models.Model(inputs, outputs, name="M3")


# ============================================================
# 2. Data + metrics
# ============================================================

def load_mnist(normalize=True):
    """Loads MNIST as (train, test), adds channel dimension."""
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
# 3. Split training loop with a single GradientTape
# ============================================================

def train_split_model(
    M1, M2, M3,
    x_train, y_train,
    x_test, y_test,
    epochs=3,
    batch_size=128,
    lr=1e-3,
):
    """
    Train split model:
        x → M1 → z_cut → M2 → z_mid → M3 → logits → loss
    using a single GradientTape to propagate gradients
    through all three models.
    """

    # Three optimizers (you could also share one)
    opt_M1 = optimizers.Adam(learning_rate=lr)
    opt_M2 = optimizers.Adam(learning_rate=lr)
    opt_M3 = optimizers.Adam(learning_rate=lr)

    loss_fn = losses.SparseCategoricalCrossentropy()
    steps_per_epoch = len(x_train) // batch_size

    print(f"Training on {len(x_train)} samples, {steps_per_epoch} steps per epoch\n")

    # Precompute variable slices to split gradients later
    vars_M1 = M1.trainable_variables
    vars_M2 = M2.trainable_variables
    vars_M3 = M3.trainable_variables
    n1, n2 = len(vars_M1), len(vars_M2)

    for epoch in range(1, epochs + 1):
        start = time.time()
        epoch_losses = []
        epoch_accs = []

        # Shuffle indices each epoch
        idx = np.random.permutation(len(x_train))
        x_train_sh = x_train[idx]
        y_train_sh = y_train[idx]

        for step in range(steps_per_epoch):
            lo = step * batch_size
            hi = lo + batch_size
            xb = tf.convert_to_tensor(x_train_sh[lo:hi], dtype=tf.float32)
            yb = tf.convert_to_tensor(y_train_sh[lo:hi], dtype=tf.int32)

            # ------------ Forward pass (explicit split) ------------
            with tf.GradientTape() as tape:
                # M1 forward
                z_cut = M1(xb, training=True)   # [B, 128]
                # M2 forward
                z_mid = M2(z_cut, training=True)  # [B, 64]
                # M3 forward
                logits = M3(z_mid, training=True)  # [B, 10]

                loss_value = loss_fn(yb, logits)

            # ------------ Backward pass (one tape) ------------
            all_vars = vars_M1 + vars_M2 + vars_M3
            grads = tape.gradient(loss_value, all_vars)

            # Split gradients back into chunks for each model
            grads_M1 = grads[:n1]
            grads_M2 = grads[n1:n1 + n2]
            grads_M3 = grads[n1 + n2:]

            # Apply updates
            opt_M1.apply_gradients(zip(grads_M1, vars_M1))
            opt_M2.apply_gradients(zip(grads_M2, vars_M2))
            opt_M3.apply_gradients(zip(grads_M3, vars_M3))

            # ------------ Metrics ------------
            acc_batch = batch_accuracy(yb.numpy(), logits.numpy())
            epoch_losses.append(float(loss_value.numpy()))
            epoch_accs.append(acc_batch)

            if step % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} Step {step}/{steps_per_epoch} "
                    f"Loss={epoch_losses[-1]:.4f} Acc={acc_batch:.4f}"
                )

        # ------------ Epoch summary ------------
        mean_loss = float(np.mean(epoch_losses))
        mean_acc = float(np.mean(epoch_accs))
        elapsed = time.time() - start
        print(
            f"\nEpoch {epoch} finished in {elapsed:.1f}s "
            f"→ Loss={mean_loss:.4f} Acc={mean_acc:.4f}\n"
        )

    # ============================================================
    # Final evaluation
    # ============================================================
    print("Evaluating on test set...")
    z1 = M1.predict(x_test, batch_size=batch_size, verbose=0)
    z2 = M2.predict(z1, batch_size=batch_size, verbose=0)
    y_pred = M3.predict(z2, batch_size=batch_size, verbose=0)
    test_acc = batch_accuracy(y_test, y_pred)
    print(f"\n✅ Final test accuracy: {test_acc:.4f}")


# ============================================================
# 4. Main
# ============================================================

def main():
    (x_train, y_train), (x_test, y_test) = load_mnist()

    M1 = build_M1()
    M2 = build_M2(input_dim=128)
    M3 = build_M3(input_dim=64)

    print(M1.summary())
    print(M2.summary())
    print(M3.summary())

    train_split_model(
        M1, M2, M3,
        x_train, y_train,
        x_test, y_test,
        epochs=3,
        batch_size=128,
        lr=1e-3,
    )


if __name__ == "__main__":
    main()
