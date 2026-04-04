from tensorflow import keras
from keras import layers, models


def build_m1_tiny_cnn(input_shape=(28, 28, 1)) -> keras.Model:
    """Tiny CNN feature extractor (lightweight)."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    return models.Model(inputs, x, name="M1_tiny_cnn")


def build_m2_tiny_cnn(input_dim=128) -> keras.Model:
    """Tiny M2 block."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    return models.Model(inputs, x, name="M2_tiny_cnn")


def build_m3_tiny_cnn(input_dim=64, num_classes=10) -> keras.Model:
    """Tiny classifier head."""
    inputs = layers.Input(shape=(input_dim,))
    outputs = layers.Dense(num_classes, activation="softmax")(inputs)
    return models.Model(inputs, outputs, name="M3_tiny_cnn")
