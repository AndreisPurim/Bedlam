#!/usr/bin/env python3
"""Utilities for toy PIR and clue handling."""

import hashlib
import json
import secrets
import time
from typing import List

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

try:
    from phe import paillier
except Exception:
    paillier = None


DEFAULT_MARKER = b"PIR0"


def derive_key(passphrase: str | None) -> bytes | None:
    if not passphrase:
        return None
    return hashlib.sha256(passphrase.encode("utf-8")).digest()


def derive_epoch_key(key: bytes, epoch: int) -> bytes:
    h = hashlib.sha256()
    h.update(key)
    h.update(epoch.to_bytes(8, "big"))
    return h.digest()


def encrypt_clue(marker: bytes, key_str: str, rotation_seconds: int = 0) -> bytes:
    key = derive_key(key_str)
    if key is None:
        return marker
    if rotation_seconds and rotation_seconds > 0:
        epoch = int(time.time()) // rotation_seconds
        key = derive_epoch_key(key, epoch)
    nonce = secrets.token_bytes(12)
    ct = AESGCM(key).encrypt(nonce, marker, None)
    return nonce + ct


def _decrypt_aead(blob: bytes, key: bytes) -> bytes:
    if len(blob) < 12:
        raise ValueError("Ciphertext too short for nonce.")
    nonce = blob[:12]
    ct = blob[12:]
    return AESGCM(key).decrypt(nonce, ct, None)


def clue_matches(
    clue: bytes,
    key_str: str,
    rotation_seconds: int = 0,
    rotation_grace: int = 1,
    marker: bytes = DEFAULT_MARKER,
) -> bool:
    key = derive_key(key_str)
    if key is None:
        return clue == marker
    if not rotation_seconds or rotation_seconds <= 0:
        try:
            return _decrypt_aead(clue, key) == marker
        except Exception:
            return False

    epoch_now = int(time.time()) // rotation_seconds
    grace = max(0, int(rotation_grace))
    for offset in range(grace + 1):
        try_epoch = epoch_now - offset
        if try_epoch < 0:
            continue
        try:
            trial_key = derive_epoch_key(key, try_epoch)
            if _decrypt_aead(clue, trial_key) == marker:
                return True
        except Exception:
            continue
    return False


# ---------------- Paillier helpers ----------------


def public_key_to_bytes(public_key) -> bytes:
    return str(public_key.n).encode("ascii")


def public_key_from_bytes(blob: bytes):
    if paillier is None:
        raise RuntimeError("phe is not installed")
    n = int(blob.decode("ascii"))
    return paillier.PaillierPublicKey(n)


def enc_number_to_bytes(enc) -> bytes:
    data = {
        "c": str(enc.ciphertext()),
        "e": int(enc.exponent),
    }
    return json.dumps(data, separators=(",", ":")).encode("ascii")


def enc_number_from_bytes(public_key, blob: bytes):
    if paillier is None:
        raise RuntimeError("phe is not installed")
    data = json.loads(blob.decode("ascii"))
    return paillier.EncryptedNumber(public_key, int(data["c"]), int(data["e"]))


def bytes_to_chunks(payload: bytes, chunk_size: int, total_len: int | None = None) -> List[int]:
    if total_len is None:
        total_len = len(payload)
    if len(payload) < total_len:
        payload = payload + b"\\x00" * (total_len - len(payload))
    chunks = []
    for i in range(0, total_len, chunk_size):
        chunk = payload[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + b"\\x00" * (chunk_size - len(chunk))
        chunks.append(int.from_bytes(chunk, "big"))
    return chunks


def chunks_to_bytes(chunks: List[int], chunk_size: int, total_len: int) -> bytes:
    out = bytearray()
    base = 1 << (8 * chunk_size)
    for val in chunks:
        v = int(val) % base
        out.extend(v.to_bytes(chunk_size, "big"))
    return bytes(out[:total_len])
