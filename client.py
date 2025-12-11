#!/usr/bin/env python3
"""
client.py — Split Learning over Ray with Encrypted, Pseudonymous Board

Topology:

  [PeerM1M3]  <-->  [Board]  <-->  [PeerM2]

Privacy-ish features:

- Board only stores:
    { msg_id, sender, receiver, payload (ciphertext), timestamp }
  No 'kind', no 'session_id' in the clear.

- Payload is an AES-GCM encrypted envelope:
    header = {
        "op": "FWD_REQ" | "FWD_RES" | "BWD_REQ" | "BWD_RES" | "INFER_REQ" | "INFER_RES",
        "session": "<random session id>",
        "sender": "<pseudonym or M2 name>",
        "tensor_len": <length of serialized tensor_bytes>,
    }

    plaintext = 4-byte header_len || header_json || tensor_bytes || padding

  Then:
    key      = SHA-256(passphrase_from_config)
    nonce    = 12 random bytes
    ct       = AESGCM(key).encrypt(nonce, plaintext, None)
    blob     = nonce || ct

- Padding:
    plaintext is padded to a multiple of the configured pad size before encryption.

- Pseudonyms:
    Each M1M3 client generates a random pseudonym per run:
        self.pseudonym = "cli_<8-hex>"
    Board and M2 only see pseudonyms, not "client_1".

- Session IDs:
    Only exist inside the encrypted header. Board never sees them.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # suppress TF INFO
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # force CPU use

import time
import uuid
import logging
from datetime import datetime
import io
import json
import hashlib
import secrets

from typing import Dict, Any

import numpy as np
from tensorflow import keras
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

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

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
# 2. Payload encoding / encrypted envelopes / padding
# ============================================================



def _derive_key(passphrase: str) -> bytes | None:
    """
    Derive a 256-bit key from a passphrase.
    If passphrase is empty, returns None (no-encryption mode).
    """
    if not passphrase:
        return None
    return hashlib.sha256(passphrase.encode("utf-8")).digest()  # 32 bytes


def tensor_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize a NumPy array to bytes (shape + dtype preserved)."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def bytes_to_tensor(b: bytes) -> np.ndarray:
    """Deserialize bytes back into a NumPy array."""
    buf = io.BytesIO(b)
    return np.load(buf, allow_pickle=False)


def _keystream(key: bytes, length: int) -> bytes:
    """
    Toy stream cipher keystream: concat SHA-256(key || counter) until we have 'length' bytes.

    NOTE: This is not production-grade crypto, but much better than a simple repeating XOR key.
    For the PoC, it gives us:
      - deterministic symmetric encryption,
      - opaque ciphertext,
      - no external library dependency.
    """
    out = bytearray()
    counter = 0
    while len(out) < length:
        h = hashlib.sha256()
        h.update(key)
        h.update(counter.to_bytes(8, "big"))
        out.extend(h.digest())
        counter += 1
    return bytes(out[:length])


def _crypt_bytes(data: bytes, key_str: str) -> bytes:
    """
    Symmetric encryption/decryption using XOR with SHA-256-based keystream.
    If key_str is empty, returns data unchanged (no-encryption mode).
    """
    key = _derive_key(key_str)
    if key is None:
        return data
    ks = _keystream(key, len(data))
    return bytes(d ^ k for d, k in zip(data, ks))


def encode_message(
    op: str,
    session: str,
    sender_pseudo: str | None,
    tensor: np.ndarray,
    key_str: str,
    target_m2: str | None = None,
    pair_id: str | None = None,
    session_done: bool = False,
    pad_multiple: int | None = None,
) -> bytes:
    """
    Build an encrypted, padded envelope:

        header = {
            "op":      "FWD_REQ" | "FWD_RES" | "BWD_REQ" | "BWD_RES"
                       | "INFER_REQ" | "INFER_RES",
            "session": "<session id>",
            "sender":  "<pseudonym or M2 name>",
            "tensor_len": <length of serialized tensor_bytes>,
        }

        plaintext = 4-byte header_len || header_json || tensor_bytes || padding

    Then:
        ciphertext = _crypt_bytes(plaintext, key_str)

    The ciphertext length is padded at the *plaintext* level to a multiple of the configured pad size
    before encryption, so the Board only sees coarse-grained sizes.
    """
    tensor_bytes = tensor_to_bytes(tensor)
    header = {
        "op": op,
        "session": session,
        "tensor_len": len(tensor_bytes),
    }
    if sender_pseudo:
        header["sender"] = sender_pseudo
    if target_m2:
        header["target_m2"] = target_m2
    if pair_id:
        header["pair_id"] = pair_id
    if session_done:
        header["session_done"] = True
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_bytes)

    base_plain = header_len.to_bytes(4, "big") + header_bytes + tensor_bytes

    if pad_multiple is None or pad_multiple <= 0:
        raise ValueError("pad_multiple must be a positive integer from config")
    # Pad plaintext to configured multiple
    pad_len = (-len(base_plain)) % pad_multiple
    if pad_len:
        base_plain += secrets.token_bytes(pad_len)

    # Encrypt (or leave as-is if key_str is empty)
    return _crypt_bytes(base_plain, key_str)


def decode_message(blob: bytes, key_str: str):
    """
    Reverse of encode_message.

    - Decrypts 'blob' with _crypt_bytes (symmetric).
    - Reads:
        header_len = first 4 bytes
        header_json = next header_len bytes
        tensor_bytes = next tensor_len bytes
      ignoring any remaining padded tail.

    Returns: (op, session, sender, tensor)
    """
    padded_plain = _crypt_bytes(blob, key_str)

    if len(padded_plain) < 4:
        raise ValueError("Plaintext too short for header length.")
    header_len = int.from_bytes(padded_plain[:4], "big")
    if len(padded_plain) < 4 + header_len:
        raise ValueError("Plaintext truncated before header.")

    header_bytes = padded_plain[4:4 + header_len]
    header = json.loads(header_bytes.decode("utf-8"))

    tensor_len = int(header["tensor_len"])
    start = 4 + header_len
    end = start + tensor_len
    if len(padded_plain) < end:
        raise ValueError("Plaintext truncated before tensor.")

    tensor_bytes = padded_plain[start:end]
    tensor = bytes_to_tensor(tensor_bytes)

    op = header["op"]
    session = header["session"]
    sender = header.get("sender")
    return op, session, sender, tensor, header


# ============================================================
# 4. Data utilities
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
# 8. Main with config + run folder + suppression flag
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
    model_arch = general.get("model_architecture", "default")
    m1_model = model_arch
    m2_model = model_arch
    m3_model = model_arch
    if "pad_multiple" not in general:
        raise ValueError("general.pad_multiple must be defined in config.yaml")
    pad_multiple = int(general["pad_multiple"])

    m2_verbose = bool(general.get("m2_verbose", False))
    m2_log_every = int(general.get("m2_log_every", 50))

    m1m3_verbose = bool(general.get("m1m3_verbose", False))
    m1m3_log_every = int(general.get("m1m3_log_every", 50))

    board_host = general.get("board_host", "localhost")
    board_port = int(general.get("board_port", 50051))
    architecture = general.get("architecture", "board-blind")

    # Delegate to other entrypoints
    if architecture in ("vanilla-split", "vanilla", "default-split"):
        from clients.vanilla_client import main as vanilla_main
        return vanilla_main(config_path)
    if architecture == "single-blind-bucket":
        from clients.bucket_client import main as bucket_main
        return bucket_main(config_path)
    if architecture == "double-blind":
        from clients.double_blind_client import main as double_blind_main
        return double_blind_main(config_path)
    if architecture == "single-blind-two-pools":
        raise ValueError("The 'single-blind-two-pools' architecture has been removed.")
    if architecture == "board-blind":
        from clients.board_blind_client import main as board_blind_main
        return board_blind_main(config_path)
    raise ValueError(f"Unknown architecture '{architecture}'. Supported: board-blind, vanilla-split, single-blind-bucket, double-blind.")

    # ---- Global logger ----
    global_logger = setup_global_logger(run_dir, log_level)
    global_logger.info(f"Run dir: {run_dir}")
    global_logger.info(
        f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"m2_verbose={m2_verbose}, m1m3_verbose={m1m3_verbose}, board_addr={board_host}:{board_port}, "
        f"architecture={architecture}, model_architecture={model_arch}"
    )
    # Save config snapshot
    try:
        import shutil
        shutil.copyfile(config_path, os.path.join(run_dir, "config_used.yaml"))
    except Exception as e:
        global_logger.warning(f"Could not save config snapshot: {e}")

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

    # ---- Build mapping from M2 name -> key ----
    m2_keys: Dict[str, str] = {}
    for m2_cfg in m2_peers_cfg:
        name = m2_cfg["name"]
        key = m2_cfg.get("key", "") or ""
        m2_keys[name] = key

    global_logger.info(f"Using external Board at {board_host}:{board_port}")

    # Shard training data across M1M3 peers
    x_shards = np.array_split(x_train, n_clients)
    y_shards = np.array_split(y_train, n_clients)

    # Create M2 peers
    m2_peers: Dict[str, Any] = {}
    for m2_cfg in m2_peers_cfg:
        name = m2_cfg["name"]
        key = m2_keys.get(name, "")
        m2_peer = PeerM2.remote(
            name=name,
            run_dir=run_dir,
            board_host=board_host,
            board_port=board_port,
            m1_model=m1_model,
            m2_model=m2_model,
            m3_model=m3_model,
            pad_multiple=pad_multiple,
            input_dim=128,
            lr=lr,
            shared_key=key,
            log_level=log_level,
            verbose=m2_verbose,
            log_every=m2_log_every,
        )
        m2_peers[name] = m2_peer
        global_logger.info(f"Spawned M2 peer: {name} (key_len={len(key)})")

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

        key = m2_keys.get(target_m2, "")
        client = PeerM1M3.remote(
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
            pad_multiple=pad_multiple,
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
        global_logger.info(
            f"Spawned M1M3 peer: {name} → bound to M2 peer: {target_m2} (key_len={len(key)})"
        )

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
