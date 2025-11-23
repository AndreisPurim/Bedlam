from tensorflow import keras
from keras import layers, models


def build_m1_default(input_shape=(28, 28, 1)) -> keras.Model:
    """Early feature extractor (default)."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    return models.Model(inputs, x, name="M1")


def build_m2_default(input_dim=128) -> keras.Model:
    """Middle model (default)."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    return models.Model(inputs, x, name="M2")


def build_m3_default(input_dim=64, num_classes=10) -> keras.Model:
    """Classifier head (default)."""
    inputs = layers.Input(shape=(input_dim,))
    outputs = layers.Dense(num_classes, activation="softmax")(inputs)
    return models.Model(inputs, outputs, name="M3")


# ------------------------------------------------------------
# Variant 1: compact (lighter than default)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Variant 1: MLP (no convs, very different footprint)
# ------------------------------------------------------------

def build_m1_mlp(input_shape=(28, 28, 1)) -> keras.Model:
    """Flatten + MLP to 128-dim; useful as a lightweight alternative."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    return models.Model(inputs, x, name="M1_mlp")


def build_m2_mlp(input_dim=128) -> keras.Model:
    """Two-layer MLP to 64-dim."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    return models.Model(inputs, x, name="M2_mlp")


def build_m3_mlp(input_dim=64, num_classes=10) -> keras.Model:
    """MLP head with hidden layer and dropout."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="M3_mlp")


# ------------------------------------------------------------
# Variant 2: deeper conv (more feature capacity)
# ------------------------------------------------------------

def build_m1_deep(input_shape=(28, 28, 1)) -> keras.Model:
    """Deeper conv stack to 128-dim."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    return models.Model(inputs, x, name="M1_deep")


def build_m2_deep(input_dim=128) -> keras.Model:
    """M2 with skip-style fusion (two layers + residual add)."""
    inputs = layers.Input(shape=(input_dim,))
    h1 = layers.Dense(96, activation="relu")(inputs)
    h2 = layers.Dense(64, activation="relu")(h1)
    # project skip to 64 and add
    skip = layers.Dense(64, activation="linear")(inputs)
    out = layers.Activation("relu")(layers.Add()([h2, skip]))
    return models.Model(inputs, out, name="M2_deep")


def build_m3_deep(input_dim=64, num_classes=10) -> keras.Model:
    """Head with two hidden layers and dropout."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="M3_deep")


# ------------------------------------------------------------
# Variant 3: lightweight ResNet-ish
# ------------------------------------------------------------

def _res_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def build_m1_resnet_lite(input_shape=(28, 28, 1)) -> keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = _res_block(x, 16, stride=1)
    x = _res_block(x, 32, stride=2)  # downsample
    x = _res_block(x, 32, stride=1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    return models.Model(inputs, x, name="M1_resnet_lite")


def build_m2_resnet_lite(input_dim=128) -> keras.Model:
    inputs = layers.Input(shape=(input_dim,))
    h1 = layers.Dense(96, activation="relu")(inputs)
    h2 = layers.Dense(64, activation="relu")(h1)
    skip = layers.Dense(64, activation="linear")(inputs)
    out = layers.Activation("relu")(layers.Add()([h2, skip]))
    return models.Model(inputs, out, name="M2_resnet_lite")


def build_m3_resnet_lite(input_dim=64, num_classes=10) -> keras.Model:
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="M3_resnet_lite")
