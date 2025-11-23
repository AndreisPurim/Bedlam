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
    plaintext is padded to a multiple of PAD_MULTIPLE bytes before encryption.

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
import grpc
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, losses, optimizers
import ray
import yaml

import board_pb2
import board_pb2_grpc
from models.factory import build_split_models

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

PAD_MULTIPLE = 1024  # pad plaintext to a multiple of this (bytes) before encryption

# Audience buckets for single-blind-two-pools pooled delivery
AUDIENCE_TO_M2 = "to_m2"
AUDIENCE_TO_CLIENTS = "to_clients"


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

    The ciphertext length is padded at the *plaintext* level to a multiple of PAD_MULTIPLE
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
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_bytes)

    base_plain = header_len.to_bytes(4, "big") + header_bytes + tensor_bytes

    # Pad plaintext to multiple of PAD_MULTIPLE
    pad_len = (-len(base_plain)) % PAD_MULTIPLE
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
# 5. gRPC Board client (for external board_server.py)
# ============================================================


class GrpcBoardClient:
    """
    Thin client around the gRPC BoardService to match the legacy Board API.
    Returns/accepts the same dict structure the Ray Board used.
    """

    def __init__(self, host: str, port: int, logger: logging.Logger | None = None):
        self.host = host
        self.port = port
        self.logger = logger or logging.getLogger("GrpcBoardClient")
        grpc_opts = [
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(f"{host}:{port}", options=grpc_opts)
        self.stub = board_pb2_grpc.BoardServiceStub(self.channel)
        self.logger.info("Connected GrpcBoardClient to %s:%d", host, port)

    def post_message(self, sender: str, receiver: str = "", payload: bytes = b"", audience: str | None = None) -> str:
        req = board_pb2.PostMessageRequest(sender=sender, receiver=receiver, payload=payload, audience=audience or "")
        resp = self.stub.PostMessage(req)
        return resp.msg_id

    def poll_message(self, receiver: str):
        req = board_pb2.PollMessageRequest(receiver=receiver)
        resp = self.stub.PollMessage(req)
        if not resp.has_message:
            return None
        msg = resp.message
        return {
            "msg_id": msg.msg_id,
            "sender": msg.sender,
            "receiver": msg.receiver,
            "payload": bytes(msg.payload),
            "timestamp": msg.timestamp_ms / 1000.0,
        }

    def poll_pool(self, audience: str, limit_count: int | None = None):
        req = board_pb2.PollPoolRequest(audience=audience)
        resp = self.stub.PollPool(req)
        return [
            {
                "msg_id": m.msg_id,
                "sender": m.sender,
                "receiver": m.receiver,
                "payload": bytes(m.payload),
                "timestamp": m.timestamp_ms / 1000.0,
            }
            for m in resp.messages
        ]

    def ack_message(self, msg_id: str, audience: str | None = None) -> bool:
        req = board_pb2.AckMessageRequest(msg_id=msg_id, audience=audience or "")
        resp = self.stub.AckMessage(req)
        return resp.removed


# ============================================================
# 6. Ray peer: M2 holder (middle model)
# ============================================================

@ray.remote
class PeerM2:
    """
    Peer that holds M2 and performs its part of the computation.

    It does NOT get called directly by clients.
    Instead, it runs a loop where it polls the Board for messages addressed
    to itself, decrypts them, inspects 'op', and acts accordingly:

      - op="FWD_REQ"   -> compute z_mid, cache tape, send "FWD_RES"
      - op="BWD_REQ"   -> use cached tape, update M2, send "BWD_RES"
      - op="INFER_REQ" -> forward-only, send "INFER_RES"

    All payloads are AES-GCM encrypted envelopes with shared_key.
    """

    def __init__(
        self,
        name: str,
        run_dir: str,
        board_host: str,
        board_port: int,
        architecture: str,
        m1_model: str,
        m2_model: str,
        m3_model: str,
        input_dim: int = 128,
        lr: float = 1e-3,
        shared_key: str | None = None,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
    ):
        self.name = name
        self.logger = setup_peer_logger(name, run_dir, log_level)
        self.board_client = GrpcBoardClient(board_host, board_port, logger=self.logger)
        self.architecture = architecture
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
        self._sessions: Dict[str, tuple] = {}  # session_id -> (tape, z_cut, z_mid)

        self.shared_key = shared_key or ""
        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0
        self._seen_msg_ids: set[str] = set()  # dedup when pooling

        self.logger.info(
            f"Initialized PeerM2 with input_dim={input_dim}, lr={lr}, "
            f"verbose={self.verbose}, log_every={self.log_every}, "
            f"shared_key_len={len(self.shared_key)} board={board_host}:{board_port} "
            f"models(m1/m2/m3)={self.m1_model}/{self.m2_model}/{self.m3_model}"
        )

    def run(self):
        """Main loop: process messages from the Board."""
        self.logger.info("PeerM2.run() loop started.")
        while True:
            if self.architecture == "board-blind":
                msgs = []
                msg = self.board_client.poll_message(receiver=self.name)
                if msg:
                    msgs.append(msg)
            else:
                msgs = self.board_client.poll_pool(AUDIENCE_TO_M2)

            if not msgs:
                time.sleep(0.01)
                continue

            for msg in msgs:
                msg_id = msg["msg_id"]
                if msg_id in self._seen_msg_ids:
                    continue
                self._seen_msg_ids.add(msg_id)

                try:
                    op, session, sender_pseudo, tensor, header = decode_message(
                        msg["payload"], self.shared_key
                    )
                except Exception as e:
                    if self.architecture == "single-blind-two-pools":
                        continue  # likely not intended for this peer/key
                    self.logger.error(f"Failed to decode message: {e}")
                    continue

                if self.architecture == "single-blind-two-pools":
                    target = header.get("target_m2")
                    if target and target != self.name:
                        continue
                    sender_pseudo = sender_pseudo or "unknown"

                handled = False
                if op == "FWD_REQ":
                    self._handle_forward(session, sender_pseudo, tensor)
                    handled = True
                elif op == "BWD_REQ":
                    self._handle_backward(session, sender_pseudo, tensor)
                    handled = True
                elif op == "INFER_REQ":
                    self._handle_infer(session, sender_pseudo, tensor)
                    handled = True
                else:
                    self.logger.warning(
                        f"Unknown op '{op}' in session={session} from={sender_pseudo}"
                    )

                if handled and self.architecture == "single-blind-two-pools":
                    # Acknowledge consumption of this request so Board can shrink the pool.
                    self.board_client.ack_message(msg_id=msg_id, audience=AUDIENCE_TO_M2)

    # ---------------- internal handlers ----------------

    def _handle_forward(self, session_id: str, sender_pseudo: str, z_cut_np: np.ndarray):
        self._fwd_count += 1

        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z_cut)
            z_mid = self.M2(z_cut, training=True)

        self._sessions[session_id] = (tape, z_cut, z_mid)
        z_mid_np = z_mid.numpy()

        payload = encode_message("FWD_RES", session_id, self.name, z_mid_np, self.shared_key, target_m2=self.name)
        if self.architecture == "board-blind":
            self.board_client.post_message(
                sender=self.name,
                receiver=sender_pseudo,
                payload=payload,
            )
        else:
            self.board_client.post_message(
                sender=self.name,
                receiver="",
                audience=AUDIENCE_TO_CLIENTS,
                payload=payload,
            )

        if self.verbose and (self._fwd_count % self.log_every == 0):
            bs = z_cut_np.shape[0]
            feat = z_cut_np.shape[1] if z_cut_np.ndim > 1 else 1
            self.logger.info(
                f"[FWD #{self._fwd_count}] session={session_id} from_pseudo={sender_pseudo} "
                f"batch={bs} feat_dim={feat}"
            )

    def _handle_backward(self, session_id: str, sender_pseudo: str, dL_dz_mid_np: np.ndarray):
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

        payload = encode_message("BWD_RES", session_id, self.name, dL_dz_cut_np, self.shared_key, target_m2=self.name)
        if self.architecture == "board-blind":
            self.board_client.post_message(
                sender=self.name,
                receiver=sender_pseudo,
                payload=payload,
            )
        else:
            self.board_client.post_message(
                sender=self.name,
                receiver="",
                audience=AUDIENCE_TO_CLIENTS,
                payload=payload,
            )

        if self.verbose and (self._bwd_count % self.log_every == 0):
            grad_norm = tf.linalg.global_norm(grads_M2).numpy()
            dzcut_norm = tf.linalg.global_norm([dL_dz_cut]).numpy()
            self.logger.info(
                f"[BWD #{self._bwd_count}] session={session_id} from_pseudo={sender_pseudo} "
                f"grad_norm(M2)={grad_norm:.4f} grad_norm(dL/dz_cut)={dzcut_norm:.4f}"
            )

    def _handle_infer(self, session_id: str, sender_pseudo: str, z_cut_np: np.ndarray):
        self._infer_count += 1

        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        z_mid = self.M2(z_cut, training=False)
        z_mid_np = z_mid.numpy()

        payload = encode_message("INFER_RES", session_id, self.name, z_mid_np, self.shared_key, target_m2=self.name)
        if self.architecture == "board-blind":
            self.board_client.post_message(
                sender=self.name,
                receiver=sender_pseudo,
                payload=payload,
            )
        else:
            self.board_client.post_message(
                sender=self.name,
                receiver="",
                audience=AUDIENCE_TO_CLIENTS,
                payload=payload,
            )

        if self.verbose and (self._infer_count % self.log_every == 0):
            bs = z_cut_np.shape[0]
            self.logger.info(
                f"[INFER #{self._infer_count}] session={session_id} from_pseudo={sender_pseudo} "
                f"batch={bs}"
            )


# ============================================================
# 7. Ray peer: M1+M3 client
# ============================================================

@ray.remote
class PeerM1M3:
    """
    A split-learning client peer (M1 + M3).

    - Holds M1 and M3.
    - Holds its own local training shard.
    - Knows:
        - the Board actor
        - the target M2 peer name (actor name)
        - the shared_key used to encode/decode envelopes

    Privacy:
    - Each client gets a random pseudonym per run:
        self.pseudonym = "cli_<8-hex>"
    - Board and M2 only see pseudonyms, never "client_1".
    """

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
        architecture: str,
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
        self.board_client = GrpcBoardClient(board_host, board_port, logger=self.logger)
        self.architecture = architecture
        self.m1_model = m1_model
        self.m2_model = m2_model
        self.m3_model = m3_model
        self.target_m2 = target_m2

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
        self._seen_msg_ids: set[str] = set()  # dedup when pooling

        # per-run pseudonym for this client
        self.pseudonym = f"cli_{uuid.uuid4().hex[:8]}"

        # metrics CSV
        self.metrics_path = os.path.join(run_dir, f"metrics_{self.actor_name}.csv")
        with open(self.metrics_path, "w") as f:
            f.write("epoch,step,loss,acc\n")

        self.logger.info(
            f"Initialized PeerM1M3 actor_name={self.actor_name} pseudonym={self.pseudonym} "
            f"epochs={epochs} batch_size={batch_size} lr={lr} "
            f"target_m2={target_m2} verbose={self.verbose} log_every={self.log_every} "
            f"shared_key_len={len(self.shared_key)} board={board_host}:{board_port} "
            f"models(m1/m2/m3)={self.m1_model}/{self.m2_model}/{self.m3_model}"
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

                session_id = f"{self.actor_name}-train-{epoch}-{step}-{uuid.uuid4().hex}"

                # ----- send FWD_REQ envelope -----
                self._fwd_count += 1
                if self.architecture == "board-blind":
                    payload = encode_message("FWD_REQ", session_id, self.pseudonym, z_cut_np, self.shared_key)
                    self.board_client.post_message(
                        sender=self.pseudonym,     # Board sees pseudonym
                        receiver=self.target_m2,   # M2 actor name
                        payload=payload,
                    )
                else:
                    payload = encode_message(
                        "FWD_REQ",
                        session_id,
                        sender_pseudo=None,
                        tensor=z_cut_np,
                        key_str=self.shared_key,
                        target_m2=self.target_m2,
                    )
                    self.board_client.post_message(
                        sender="",
                        receiver="",
                        audience=AUDIENCE_TO_M2,
                        payload=payload,
                    )
                if self.verbose and (self._fwd_count % self.log_every == 0):
                    self.logger.info(
                        f"[FWD_REQ #{self._fwd_count}] session={session_id} "
                        f"to={self.target_m2} batch={z_cut_np.shape[0]}"
                    )

                # ----- wait for FWD_RES -----
                z_mid_np = None
                while z_mid_np is None:
                    if self.architecture == "board-blind":
                        msgs = []
                        msg = self.board_client.poll_message(
                            receiver=self.pseudonym,   # responses addressed to pseudonym
                        )
                        if msg:
                            msgs.append(msg)
                    else:
                        msgs = self.board_client.poll_pool(AUDIENCE_TO_CLIENTS)
                    if not msgs:
                        time.sleep(0.01)
                        continue
                    for msg in msgs:
                        msg_id = msg["msg_id"]
                        if msg_id in self._seen_msg_ids:
                            continue
                        self._seen_msg_ids.add(msg_id)
                        try:
                            op, sess, sender_name, tensor, header = decode_message(msg["payload"], self.shared_key)
                        except Exception as e:
                            if self.architecture == "single-blind-two-pools":
                                continue  # not for this client/key
                            self.logger.error(f"Failed to decode FWD_RES: {e}")
                            continue
                        if op != "FWD_RES" or sess != session_id:
                            continue
                        z_mid_np = tensor
                        if self.verbose and (self._fwd_count % self.log_every == 0):
                            self.logger.info(
                                f"[FWD_RES #{self._fwd_count}] session={session_id} "
                                f"from={sender_name} shape={z_mid_np.shape}"
                            )
                        # acknowledge to shrink pool
                        if self.architecture == "single-blind-two-pools":
                            self.board_client.ack_message(msg_id=msg_id, audience=AUDIENCE_TO_CLIENTS)
                        if self.architecture == "single-blind-two-pools":
                            self.board_client.ack_message(msg_id=msg_id, audience=AUDIENCE_TO_CLIENTS)
                        break
                    if z_mid_np is None:
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

                # ----- send BWD_REQ -----
                self._bwd_count += 1
                dL_dz_mid_np = dL_dz_mid.numpy()
                if self.architecture == "board-blind":
                    payload = encode_message("BWD_REQ", session_id, self.pseudonym, dL_dz_mid_np, self.shared_key)
                    self.board_client.post_message(
                        sender=self.pseudonym,
                        receiver=self.target_m2,
                        payload=payload,
                    )
                else:
                    payload = encode_message(
                        "BWD_REQ",
                        session_id,
                        sender_pseudo=None,
                        tensor=dL_dz_mid_np,
                        key_str=self.shared_key,
                        target_m2=self.target_m2,
                    )
                    self.board_client.post_message(
                        sender="",
                        receiver="",
                        audience=AUDIENCE_TO_M2,
                        payload=payload,
                    )
                if self.verbose and (self._bwd_count % self.log_every == 0):
                    self.logger.info(
                        f"[BWD_REQ #{self._bwd_count}] session={session_id} "
                        f"to={self.target_m2}"
                    )

                # ----- wait for BWD_RES -----
                dL_dz_cut_np = None
                while dL_dz_cut_np is None:
                    if self.architecture == "board-blind":
                        msgs = []
                        msg = self.board_client.poll_message(
                            receiver=self.pseudonym,
                        )
                        if msg:
                            msgs.append(msg)
                    else:
                        msgs = self.board_client.poll_pool(AUDIENCE_TO_CLIENTS)
                    if not msgs:
                        time.sleep(0.01)
                        continue
                    for msg in msgs:
                        msg_id = msg["msg_id"]
                        if msg_id in self._seen_msg_ids:
                            continue
                        self._seen_msg_ids.add(msg_id)
                        try:
                            op, sess, sender_name, tensor, header = decode_message(msg["payload"], self.shared_key)
                        except Exception as e:
                            if self.architecture == "single-blind-two-pools":
                                continue  # not for this client/key
                            self.logger.error(f"Failed to decode BWD_RES: {e}")
                            continue
                        if op != "BWD_RES" or sess != session_id:
                            continue
                        dL_dz_cut_np = tensor
                        if self.verbose and (self._bwd_count % self.log_every == 0):
                            self.logger.info(
                                f"[BWD_RES #{self._bwd_count}] session={session_id} "
                                f"from={sender_name}"
                            )
                        break
                    if dL_dz_cut_np is None:
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
            self.logger.info(
                f"Epoch {epoch} done in {elapsed:.1f}s "
                f"→ Loss={mean_loss:.4f} Acc={mean_acc:.4f}"
            )

        return f"{self.actor_name} training finished."

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

            session_id = f"{self.actor_name}-eval-{i}-{uuid.uuid4().hex}"

            # send INFER_REQ
            self._infer_count += 1
            if self.architecture == "board-blind":
                payload = encode_message("INFER_REQ", session_id, self.pseudonym, z_cut_np, self.shared_key)
                self.board_client.post_message(
                    sender=self.pseudonym,
                    receiver=self.target_m2,
                    payload=payload,
                )
            else:
                payload = encode_message(
                    "INFER_REQ",
                    session_id,
                    sender_pseudo=None,
                    tensor=z_cut_np,
                    key_str=self.shared_key,
                    target_m2=self.target_m2,
                )
                self.board_client.post_message(
                    sender="",
                    receiver="",
                    audience=AUDIENCE_TO_M2,
                    payload=payload,
                )
            if self.verbose and (self._infer_count % self.log_every == 0):
                self.logger.info(
                    f"[INFER_REQ #{self._infer_count}] session={session_id} "
                    f"to={self.target_m2} batch={z_cut_np.shape[0]}"
                )

            # wait for INFER_RES
            z_mid_np = None
            while z_mid_np is None:
                if self.architecture == "board-blind":
                    msgs = []
                    msg = self.board_client.poll_message(
                        receiver=self.pseudonym,
                    )
                    if msg:
                        msgs.append(msg)
                else:
                    msgs = self.board_client.poll_pool(AUDIENCE_TO_CLIENTS)
                if not msgs:
                    time.sleep(0.01)
                    continue
                for msg in msgs:
                    msg_id = msg["msg_id"]
                    if msg_id in self._seen_msg_ids:
                        continue
                    self._seen_msg_ids.add(msg_id)
                    try:
                        op, sess, sender_name, tensor, header = decode_message(msg["payload"], self.shared_key)
                    except Exception as e:
                        if self.architecture == "single-blind-two-pools":
                            continue  # not for this client/key
                        self.logger.error(f"Failed to decode INFER_RES: {e}")
                        continue
                    if op != "INFER_RES" or sess != session_id:
                        continue
                    z_mid_np = tensor
                    if self.verbose and (self._infer_count % self.log_every == 0):
                        self.logger.info(
                            f"[INFER_RES #{self._infer_count}] session={session_id} "
                            f"from={sender_name} shape={z_mid_np.shape}"
                        )
                    if self.architecture == "single-blind-two-pools":
                        self.board_client.ack_message(msg_id=msg_id, audience=AUDIENCE_TO_CLIENTS)
                    break
                if z_mid_np is None:
                    time.sleep(0.01)

            z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)
            logits = self.M3(z_mid, training=False)
            all_logits.append(logits.numpy())

        logits_full = np.concatenate(all_logits, axis=0)[:n]
        acc = batch_accuracy(self.y_test, logits_full)
        self.logger.info(f"Test accuracy: {acc:.4f}")
        return float(acc)


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
    m1_model = general.get("m1_model", "default")
    m2_model = general.get("m2_model", "default")
    m3_model = general.get("m3_model", "default")

    m2_verbose = bool(general.get("m2_verbose", False))
    m2_log_every = int(general.get("m2_log_every", 50))

    m1m3_verbose = bool(general.get("m1m3_verbose", False))
    m1m3_log_every = int(general.get("m1m3_log_every", 50))

    board_host = general.get("board_host", "localhost")
    board_port = int(general.get("board_port", 50051))
    architecture = general.get("architecture", "board-blind")

    # Delegate to bucket mode entrypoint
    if architecture == "single-blind-bucket":
        from bucket_client import main as bucket_main
        return bucket_main(config_path)

    # ---- Global logger ----
    global_logger = setup_global_logger(run_dir, log_level)
    global_logger.info(f"Run dir: {run_dir}")
    global_logger.info(
        f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"m2_verbose={m2_verbose}, m1m3_verbose={m1m3_verbose}, board_addr={board_host}:{board_port}, "
        f"architecture={architecture}, model_architecture={model_arch}"
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
            architecture=architecture,
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
            architecture=architecture,
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
