#!/usr/bin/env python3
"""
double_blind.py -- Double-blind privacy architecture.

This mode:
 - Uses an external pairing server (separate process) to exchange public keys
   between M1M3 and M2 peers without revealing identities.
 - Derives a symmetric key via Diffie-Hellman; both peers then encrypt all
   traffic through the bucket-based board. The board remains blind.
 - The pairing server only sees availability and request messages containing
   public keys. Once a match is made, the queued request is dropped.
 - M2 peers decrypt every bucket they see; if decryption works with their
   current session key, they process the message. A SESSION_DONE message makes
   the M2 re-register as available with a fresh keypair.

Board server and pairing server are separate processes. The client workflow is:
   1) Talk to pairing server to get a partner/public key; derive symmetric key.
   2) Run training/inference via the board (bucket strategy), encrypting with
      the derived symmetric key.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import base64
import json
import logging
import time
import uuid
import hashlib
import random
from datetime import datetime
from typing import Dict, Any

import numpy as np
import ray
import grpc
import tensorflow as tf
from keras import losses, optimizers
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import dh

# gRPC JSON helpers used by PairingClient


def _serialize(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def _deserialize(data: bytes) -> dict:
    if not data:
        return {}
    return json.loads(data.decode("utf-8"))


from client import (
    encode_message,
    decode_message,
    load_dataset,
    batch_accuracy,
    setup_global_logger,
    setup_peer_logger,
)
from models.factory import build_split_models
from peers.base import BaseBucketBoardClient, BaseBucketM1M3Peer, BaseBucketM2Peer, BasePIRBoardClient, stratified_split

# ============================================================
# Pairing client (gRPC JSON)
# ============================================================


class PairingClient:
    def __init__(self, host: str, port: int, logger: logging.Logger):
        self.base = f"{host}:{port}"
        self.logger = logger
        opts = [
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(self.base, options=opts)

    def _call(self, method: str, payload: dict):
        stub = self.channel.unary_unary(
            f"/pairing.PairingService/{method}",
            request_serializer=_serialize,
            response_deserializer=_deserialize,
        )
        return stub(payload)

    def get_dh_params(self):
        data = self._call("DhParams", {})
        return data["p"], data["g"]

    def request_pair(self, client_id: str, public_key: bytes):
        data = self._call(
            "Request",
            {
                "client_id": client_id,
                "public_key": base64.b64encode(public_key).decode("ascii"),
            },
        )
        return self._decode_assignment(data)

    def poll_assignment_client(self, client_id: str):
        data = self._call("PollAssignment", {"client_id": client_id})
        return self._decode_assignment(data)

    def register_m2(self, m2_id: str, public_key: bytes):
        data = self._call(
            "RegisterM2",
            {
                "m2_id": m2_id,
                "public_key": base64.b64encode(public_key).decode("ascii"),
            },
        )
        return self._decode_assignment(data)

    def poll_assignment_m2(self, m2_id: str):
        data = self._call("PollAssignmentM2", {"m2_id": m2_id})
        return self._decode_assignment(data)

    @staticmethod
    def _decode_assignment(data: dict):
        if not data.get("assigned"):
            return None
        peer_pk_b64 = data.get("peer_public_key", "")
        if not peer_pk_b64:
            return None
        return base64.b64decode(peer_pk_b64)


# ============================================================
# Crypto helpers
# ============================================================


def _serialize_public_key(public_key) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _load_public_key(public_key_bytes: bytes):
    return serialization.load_pem_public_key(public_key_bytes)


def _derive_shared_key_hex(private_key, peer_public_bytes: bytes) -> str:
    peer_public = _load_public_key(peer_public_bytes)
    shared_secret = private_key.exchange(peer_public)
    return hashlib.sha256(shared_secret).hexdigest()


# ============================================================
# Double-blind M2 peer
# ============================================================


@ray.remote
class DoubleBlindPeerM2(BaseBucketM2Peer):
    def __init__(
        self,
        name: str,
        run_dir: str,
        board_host: str,
        board_port: int,
        pairing_host: str,
        pairing_port: int,
        m1_model: str = "default",
        m2_model: str = "default",
        m3_model: str = "default",
        pad_multiple: int = 1024,
        input_dim: int = 128,
        lr: float = 1e-3,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
        key_rotation_seconds: int = 0,
        key_rotation_grace: int = 1,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        max_messages_per_poll: int = 0,
        batch_delay_ms: int = 0,
        poll_shuffle: bool = False,
        use_pir: bool = False,
        pir_chunk_size: int = 32,
        pir_clue_limit: int = 0,
    ):
        self.pad_multiple = pad_multiple
        self.key_rotation_seconds = key_rotation_seconds
        self.key_rotation_grace = key_rotation_grace
        self.max_messages_per_poll = max(0, int(max_messages_per_poll))
        self.batch_delay_ms = max(0, int(batch_delay_ms))
        self.poll_shuffle = bool(poll_shuffle)
        self.key_rotation_seconds = key_rotation_seconds
        self.key_rotation_grace = key_rotation_grace
        logger = setup_peer_logger(name, run_dir, log_level)
        self.pairing = PairingClient(pairing_host, pairing_port, logger=logger)
        board = BaseBucketBoardClient(board_host, board_port)
        pir_client = BasePIRBoardClient(board_host, board_port, logger=logger) if use_pir else None

        _, M2, _ = build_split_models(
            m1_name=m1_model,
            m2_name=m2_model,
            m3_name=m3_model,
            m2_kwargs={"input_dim": input_dim},
        )
        opt_M2 = optimizers.Adam(learning_rate=lr)

        def encode_fn(op, session, sender, tensor, reply_token=None):
            return encode_message(
                op,
                session,
                sender_pseudo=None,
                tensor=tensor,
                key_str=self._shared_key_hex or "",
                pad_multiple=self.pad_multiple,
                reply_token=reply_token,
                rotation_seconds=key_rotation_seconds,
            )

        def decode_fn(blob: bytes):
            op, session, sender, tensor, header = decode_message(
                blob,
                self._shared_key_hex or "",
                rotation_seconds=key_rotation_seconds,
                rotation_grace=key_rotation_grace,
            )
            return op, session, sender, tensor, header

        self._dh_params = None
        self._private_key = None
        self._public_key_bytes = None
        self._shared_key_hex = None

        super().__init__(
            name=name,
            board_client=board,
            model=M2,
            optimizer=opt_M2,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            logger=logger,
            target_name=None,
            verbose=verbose,
            log_every=log_every,
            replay_protection=replay_protection,
            replay_cache_size=replay_cache_size,
            replay_window_ms=replay_window_ms,
            replay_future_ms=replay_future_ms,
            send_jitter_ms=send_jitter_ms,
            poll_jitter_ms=poll_jitter_ms,
            dummy_rate=dummy_rate,
            use_pir=use_pir,
            pir_client=pir_client,
            pir_key="",
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
            pir_rotation_seconds=key_rotation_seconds,
            pir_rotation_grace=key_rotation_grace,
        )

        logger.info(
            "Initialized DoubleBlindPeerM2 board=%s:%d pairing=%s:%d pad=%d lr=%.4f",
            board_host,
            board_port,
            pairing_host,
            pairing_port,
            pad_multiple,
            lr,
        )

    # ---------------- key + pairing ----------------

    def _ensure_dh_params(self):
        if self._dh_params is None:
            p, g = self.pairing.get_dh_params()
            numbers = dh.DHParameterNumbers(p, g)
            self._dh_params = numbers.parameters()

    def _rotate_keypair(self):
        self._ensure_dh_params()
        self._private_key = self._dh_params.generate_private_key()
        self._public_key_bytes = _serialize_public_key(self._private_key.public_key())

    def _wait_for_assignment(self):
        # Drop any stale assignment from a previous session before re-registering.
        try:
            self.pairing.poll_assignment_m2(self.name)
        except Exception:
            pass
        self._rotate_keypair()
        self.logger.info("[pairing] registering availability with pairing server")
        assignment = self.pairing.register_m2(self.name, self._public_key_bytes)
        while assignment is None:
            self.logger.info("[pairing] waiting for client request...")
            time.sleep(0.05)
            assignment = self.pairing.poll_assignment_m2(self.name)
        self._shared_key_hex = _derive_shared_key_hex(self._private_key, assignment)
        if self.use_pir:
            self._set_pir_key(self._shared_key_hex)
        self.logger.info("[pairing] paired with a client; ready for session")

    # ---------------- main loop ----------------

    def run(self):
        self.logger.info("DoubleBlindPeerM2.run loop started.")
        while True:
            self._wait_for_assignment()
            self._session_loop()

    def _session_loop(self):
        while True:
            if self.use_pir:
                item = self._pir_next_payload()
                if not item:
                    self._sleep_poll()
                    continue
                idx, payload = item
                self.bytes_received += len(payload)
                try:
                    op, session, sender, tensor, header = decode_message(
                        payload,
                        self._shared_key_hex or "",
                        rotation_seconds=self.key_rotation_seconds,
                        rotation_grace=self.key_rotation_grace,
                    )
                except Exception:
                    self._pir_retry_index(idx)
                    continue
                if self._is_replay(header):
                    self._pir_mark_retrieved(idx)
                    continue

                if op == "DUMMY":
                    self._dummy_count += 1
                    self._pir_mark_retrieved(idx)
                    continue

                if header.get("session_done") or op == "SESSION_DONE":
                    self.logger.info("Session done signal received; re-registering availability")
                    self._pir_mark_retrieved(idx)
                    self._cleanup_session()
                    return

                if op.endswith("_RES"):
                    self._pir_mark_retrieved(idx)
                    continue

                reply_token = header.get("reply_token")
                if op == "FWD_REQ":
                    self._handle_forward("", session, sender, tensor, reply_token=reply_token)
                elif op == "BWD_REQ":
                    self._handle_backward("", session, sender, tensor, reply_token=reply_token)
                elif op == "INFER_REQ":
                    self._handle_infer("", session, sender, tensor, reply_token=reply_token)
                self._pir_mark_retrieved(idx)

                total_ops = self._fwd_count + self._bwd_count + self._infer_count
                if total_ops and (total_ops % self.log_every == 0):
                    self.logger.info(
                        "[bytes] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
                        self.bytes_sent,
                        self.bytes_received,
                        self._fwd_count,
                        self._bwd_count,
                        self._infer_count,
                    )
                continue
            buckets = self.board.poll_buckets()
            if not buckets:
                self._sleep_poll()
                continue
            if self.poll_shuffle and len(buckets) > 1:
                random.shuffle(buckets)
            if self.max_messages_per_poll > 0:
                buckets = buckets[: self.max_messages_per_poll]
            for b in buckets:
                bid = b["bucket_id"]
                self.bytes_received += len(b["payload"])
                try:
                    op, session, sender, tensor, header = decode_message(
                        b["payload"],
                        self._shared_key_hex or "",
                        rotation_seconds=self.key_rotation_seconds,
                        rotation_grace=self.key_rotation_grace,
                    )
                except Exception as e:
                    self.logger.debug(f"[decode-fail] bucket={bid} err={e}")
                    continue
                if self._is_replay(header):
                    continue

                if op == "DUMMY":
                    self.board.ack_bucket(bid)
                    self._dummy_count += 1
                    continue

                if header.get("session_done") or op == "SESSION_DONE":
                    self.board.ack_bucket(bid)
                    self.logger.info("Session done signal received; re-registering availability")
                    self._cleanup_session()
                    return

                reply_token = header.get("reply_token")
                if op == "FWD_REQ":
                    self._handle_forward(bid, session, sender, tensor, reply_token=reply_token)
                elif op == "BWD_REQ":
                    self._handle_backward(bid, session, sender, tensor, reply_token=reply_token)
                elif op == "INFER_REQ":
                    self._handle_infer(bid, session, sender, tensor, reply_token=reply_token)

            total_ops = self._fwd_count + self._bwd_count + self._infer_count
            if total_ops and (total_ops % self.log_every == 0):
                self.logger.info(
                    "[bytes] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
                    self.bytes_sent,
                    self.bytes_received,
                    self._fwd_count,
                    self._bwd_count,
                    self._infer_count,
                )
            if self.batch_delay_ms > 0:
                time.sleep(self.batch_delay_ms / 1000.0)

    def _cleanup_session(self):
        self._sessions.clear()
        self._shared_key_hex = None
        self._private_key = None
        self._public_key_bytes = None
        if self.use_pir:
            self._set_pir_key("")
        try:
            self.pairing.poll_assignment_m2(self.name)
        except Exception:
            pass

    # ---------------- handlers ----------------

    def _handle_forward(
        self,
        bucket_id: str,
        session_id: str,
        sender: str,
        z_cut_np: np.ndarray,
        reply_token: str | None = None,
    ):
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z_cut)
            z_mid = self.M2(z_cut, training=True)
        self._sessions[session_id] = (tape, z_cut, z_mid)
        z_mid_np = z_mid.numpy()

        self._fwd_count += 1

        resp = encode_message(
            "FWD_RES",
            session_id,
            sender_pseudo=None,
            tensor=z_mid_np,
            key_str=self._shared_key_hex,
            pad_multiple=self.pad_multiple,
            reply_token=reply_token,
            rotation_seconds=self.key_rotation_seconds,
        )
        self._sleep_send()
        if self.use_pir:
            self._pir_post(resp)
        else:
            self.board.update_bucket(bucket_id, resp)
        self.bytes_sent += len(resp)

    def _handle_backward(
        self,
        bucket_id: str,
        session_id: str,
        sender: str,
        dL_dz_mid_np: np.ndarray,
        reply_token: str | None = None,
    ):
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

        self._bwd_count += 1

        resp = encode_message(
            "BWD_RES",
            session_id,
            sender_pseudo=None,
            tensor=dL_dz_cut_np,
            key_str=self._shared_key_hex,
            pad_multiple=self.pad_multiple,
            reply_token=reply_token,
            rotation_seconds=self.key_rotation_seconds,
        )
        self._sleep_send()
        if self.use_pir:
            self._pir_post(resp)
        else:
            self.board.update_bucket(bucket_id, resp)
        self.bytes_sent += len(resp)

    def _handle_infer(
        self,
        bucket_id: str,
        session_id: str,
        sender: str,
        z_cut_np: np.ndarray,
        reply_token: str | None = None,
    ):
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        z_mid = self.M2(z_cut, training=False)
        z_mid_np = z_mid.numpy()

        self._infer_count += 1

        resp = encode_message(
            "INFER_RES",
            session_id,
            sender_pseudo=None,
            tensor=z_mid_np,
            key_str=self._shared_key_hex,
            pad_multiple=self.pad_multiple,
            reply_token=reply_token,
            rotation_seconds=self.key_rotation_seconds,
        )
        self._sleep_send()
        if self.use_pir:
            self._pir_post(resp)
        else:
            self.board.update_bucket(bucket_id, resp)
        self.bytes_sent += len(resp)


# ============================================================
# Double-blind M1M3 peer
# ============================================================


@ray.remote
class DoubleBlindPeerM1M3(BaseBucketM1M3Peer):
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
        pairing_host: str,
        pairing_port: int,
        m1_model: str = "default",
        m2_model: str = "default",
        m3_model: str = "default",
        pad_multiple: int = 1024,
        epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        max_steps_per_epoch: int = 0,
        log_level: str = "INFO",
        verbose: bool = False,
        log_every: int = 50,
        key_rotation_seconds: int = 0,
        key_rotation_grace: int = 1,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        poll_shuffle: bool = False,
        use_pir: bool = False,
        pir_chunk_size: int = 32,
        pir_clue_limit: int = 0,
    ):
        self.pad_multiple = pad_multiple
        self.poll_shuffle = bool(poll_shuffle)
        self.key_rotation_seconds = key_rotation_seconds
        self.key_rotation_grace = key_rotation_grace
        logger = setup_peer_logger(name, run_dir, log_level)
        self.pairing = PairingClient(pairing_host, pairing_port, logger=logger)
        board = BaseBucketBoardClient(board_host, board_port)
        pir_client = BasePIRBoardClient(board_host, board_port, logger=logger) if use_pir else None

        M1, _, M3 = build_split_models(
            m1_name=m1_model,
            m2_name=m2_model,
            m3_name=m3_model,
            m3_kwargs={"input_dim": 64},
        )
        opt_M1 = optimizers.Adam(learning_rate=lr)
        opt_M3 = optimizers.Adam(learning_rate=lr)
        loss_fn = losses.SparseCategoricalCrossentropy()

        def encode_fn(op, session, sender, tensor, reply_token=None):
            return encode_message(
                op,
                session,
                sender_pseudo=None,
                tensor=tensor,
                key_str=self._shared_key_hex or "",
                target_m2=None,
                pad_multiple=self.pad_multiple,
                reply_token=reply_token,
                rotation_seconds=self.key_rotation_seconds,
            )

        def decode_fn(blob: bytes):
            op, session, sender, tensor, header = decode_message(
                blob,
                self._shared_key_hex or "",
                rotation_seconds=self.key_rotation_seconds,
                rotation_grace=self.key_rotation_grace,
            )
            return op, session, sender, tensor, header

        self._dh_params = None
        self._private_key = None
        self._shared_key_hex = None

        super().__init__(
            name=name,
            run_dir=run_dir,
            board_client=board,
            M1=M1,
            M3=M3,
            opt_M1=opt_M1,
            opt_M3=opt_M3,
            loss_fn=loss_fn,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            target_m2="",
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            max_steps_per_epoch=max_steps_per_epoch,
            logger=logger,
            verbose=verbose,
            log_every=log_every,
            replay_protection=replay_protection,
            replay_cache_size=replay_cache_size,
            replay_window_ms=replay_window_ms,
            replay_future_ms=replay_future_ms,
            send_jitter_ms=send_jitter_ms,
            poll_jitter_ms=poll_jitter_ms,
            dummy_rate=dummy_rate,
            poll_shuffle=poll_shuffle,
            use_pir=use_pir,
            pir_client=pir_client,
            pir_key="",
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
            pir_rotation_seconds=self.key_rotation_seconds,
            pir_rotation_grace=self.key_rotation_grace,
        )

        self.metrics_path = os.path.join(run_dir, f"metrics_{name}.csv")
        with open(self.metrics_path, "w") as f:
            f.write("epoch,step,loss,acc\n")
        self._last_acc = None

        logger.info(
            "Initialized DoubleBlindPeerM1M3 name=%s epochs=%d batch_size=%d lr=%.4f",
            name,
            epochs,
            batch_size,
            lr,
        )

    # ---------------- pairing helpers ----------------

    def _ensure_dh_params(self):
        if self._dh_params is None:
            p, g = self.pairing.get_dh_params()
            numbers = dh.DHParameterNumbers(p, g)
            self._dh_params = numbers.parameters()

    def _reset_pair_state(self):
        self._private_key = None
        self._shared_key_hex = None
        if self.use_pir:
            self._set_pir_key("")
        try:
            # Clear any stale assignment that might still be queued for this client.
            self.pairing.poll_assignment_client(self.name)
        except Exception:
            pass

    def _request_pair(self, max_wait_sec: float = 120.0):
        """
        Poll pairing server until an M2 is assigned. Periodically refreshes the
        request (including a rotated DH key) to avoid stale queue entries.
        """
        self._ensure_dh_params()
        overall_start = time.time()
        attempt = 0
        while True:
            attempt_start = time.time()
            attempt += 1
            try:
                # Drop any leftover assignment from previous attempts before re-requesting.
                self.pairing.poll_assignment_client(self.name)
            except Exception:
                pass
            self._private_key = self._dh_params.generate_private_key()
            public_bytes = _serialize_public_key(self._private_key.public_key())
            self.logger.info("[pairing] requesting partner from pairing server (attempt=%d)", attempt)
            assignment = self.pairing.request_pair(self.name, public_bytes)
            polls = 0

            while assignment is None:
                time.sleep(0.05)
                assignment = self.pairing.poll_assignment_client(self.name)
                polls += 1

                now = time.time()
                if polls % 40 == 0:
                    self.logger.info(
                        "[pairing] still waiting for available M2... polls=%d elapsed=%.1fs",
                        polls,
                        now - attempt_start,
                    )

                if max_wait_sec and (now - attempt_start) > max_wait_sec:
                    # Refresh everything (including DH key) and try again.
                    self.logger.info(
                        "[pairing] refresh window exceeded (%.1fs); rotating key and re-queueing request",
                        max_wait_sec,
                    )
                    break

            if assignment is None:
                time.sleep(0.1)
                # Go back up to send a fresh request.
                continue

            self._shared_key_hex = _derive_shared_key_hex(self._private_key, assignment)
            if self.use_pir:
                self._set_pir_key(self._shared_key_hex)
            self.logger.info("[pairing] partner found after %.1fs; starting session", time.time() - overall_start)
            return

    # ---------------- training ----------------

    def train(self):
        self._request_pair()

        n = len(self.x_train)
        steps_per_epoch = n // self.batch_size
        if self.max_steps_per_epoch and self.max_steps_per_epoch < steps_per_epoch:
            steps_per_epoch = self.max_steps_per_epoch
        self.logger.info(f"Starting training on {n} samples, {steps_per_epoch} steps/epoch.")

        for epoch in range(1, self.epochs + 1):
            sent_start = self.bytes_sent
            recv_start = self.bytes_received
            fwd_start = self._fwd_count
            bwd_start = self._bwd_count
            infer_start = self._infer_count

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

                retries = 0
                while True:
                    bucket_id = None
                    try:
                        with tf.GradientTape(persistent=True) as tape_M1:
                            z_cut = self.M1(xb, training=True)
                        z_cut_np = z_cut.numpy()

                        session_id = f"{self.name}-train-{epoch}-{step}-{uuid.uuid4().hex}"
                        reply_token = uuid.uuid4().hex

                        self._fwd_count += 1
                        payload = encode_message(
                            "FWD_REQ",
                            session_id,
                            sender_pseudo=None,
                            tensor=z_cut_np,
                            key_str=self._shared_key_hex,
                            pad_multiple=self.pad_multiple,
                            reply_token=reply_token,
                            rotation_seconds=self.key_rotation_seconds,
                        )
                        self._sleep_send()
                        if self.use_pir:
                            bucket_id = ""
                            self._pir_post(payload)
                        else:
                            bucket_id = self.board.create_bucket(payload)
                        self.bytes_sent += len(payload)
                        self._maybe_send_dummy()

                        z_mid_np = self._wait_for_response(
                            bucket_id,
                            session_id,
                            "FWD_RES",
                            reply_token=reply_token,
                            timeout_sec=120,
                        )
                        z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)

                        with tf.GradientTape() as tape_M3:
                            tape_M3.watch(z_mid)
                            logits = self.M3(z_mid, training=True)
                            loss_value = self.loss_fn(yb, logits)

                        targets = self.M3.trainable_variables + [z_mid]
                        grads_all = tape_M3.gradient(loss_value, targets)
                        grads_M3 = grads_all[:-1]
                        dL_dz_mid = grads_all[-1]
                        self.opt_M3.apply_gradients(zip(grads_M3, self.M3.trainable_variables))

                        self._bwd_count += 1
                        dL_dz_mid_np = dL_dz_mid.numpy()
                        payload = encode_message(
                            "BWD_REQ",
                            session_id,
                            sender_pseudo=None,
                            tensor=dL_dz_mid_np,
                            key_str=self._shared_key_hex,
                            pad_multiple=self.pad_multiple,
                            reply_token=reply_token,
                            rotation_seconds=self.key_rotation_seconds,
                        )
                        self._sleep_send()
                        if self.use_pir:
                            self._pir_post(payload)
                        else:
                            self.board.update_bucket(bucket_id, payload)
                        self.bytes_sent += len(payload)
                        self._maybe_send_dummy()

                        dL_dz_cut_np = self._wait_for_response(
                            bucket_id,
                            session_id,
                            "BWD_RES",
                            reply_token=reply_token,
                            timeout_sec=120,
                        )
                        if not self.use_pir:
                            self.board.ack_bucket(bucket_id)

                        dL_dz_cut = tf.convert_to_tensor(dL_dz_cut_np, dtype=tf.float32)
                        grads_M1 = tape_M1.gradient(z_cut, self.M1.trainable_variables, output_gradients=dL_dz_cut)
                        self.opt_M1.apply_gradients(zip(grads_M1, self.M1.trainable_variables))
                        del tape_M1

                        acc_batch = batch_accuracy(yb.numpy(), logits.numpy())
                        loss_val = float(loss_value.numpy())
                        epoch_losses.append(loss_val)
                        epoch_accs.append(acc_batch)
                        with open(self.metrics_path, "a") as f:
                            f.write(f"{epoch},{step},{loss_val},{acc_batch}\n")

                        if step % 100 == 0:
                            self.logger.info(
                                f"Epoch {epoch} Step {step}/{steps_per_epoch} Loss={loss_val:.4f} Acc={acc_batch:.4f}"
                            )
                        break
                    except TimeoutError:
                        retries += 1
                        self.logger.error(
                            "[train] timeout waiting for response (step=%d epoch=%d attempt=%d); re-pairing and retrying",
                            step,
                            epoch,
                            retries,
                        )
                        if bucket_id and not self.use_pir:
                            try:
                                self.board.ack_bucket(bucket_id)
                            except Exception:
                                pass
                        self._send_session_done()
                        if retries >= 3:
                            raise
                        self._request_pair()
                        time.sleep(0.5)

            mean_loss = float(np.mean(epoch_losses))
            mean_acc = float(np.mean(epoch_accs))
            elapsed = time.time() - start
            self.logger.info(f"Epoch {epoch} done in {elapsed:.1f}s -> Loss={mean_loss:.4f} Acc={mean_acc:.4f}")
            self.logger.info(
                f"[bytes-epoch] epoch={epoch} sent={self.bytes_sent - sent_start}B recv={self.bytes_received - recv_start}B "
                f"fwd={self._fwd_count - fwd_start} bwd={self._bwd_count - bwd_start} infer={self._infer_count - infer_start}"
            )

        # Release M2 for reuse before evaluation so queued peers can pair
        self._send_session_done()
        self.logger.info(
            f"[bytes-summary] sent={self.bytes_sent}B recv={self.bytes_received}B "
            f"fwd={self._fwd_count} bwd={self._bwd_count} infer={self._infer_count}"
        )
        return f"{self.name} training finished."

    def _wait_for_response(
        self,
        bucket_id: str,
        session_id: str,
        expect_op: str,
        reply_token: str | None = None,
        timeout_sec: float | None = None,
    ):
        start_wait = time.time()
        last_log = start_wait
        decode_failures = 0
        while True:
            if self.use_pir:
                item = self._pir_next_payload()
                if not item:
                    self._sleep_poll()
                else:
                    idx, payload = item
                    self.bytes_received += len(payload)
                    try:
                        op, sess, sender, tensor, header = decode_message(
                            payload,
                            self._shared_key_hex,
                            rotation_seconds=self.key_rotation_seconds,
                            rotation_grace=self.key_rotation_grace,
                        )
                    except Exception:
                        decode_failures += 1
                        if decode_failures <= 3:
                            self.logger.info("[decode-fail] pir err")
                        self._pir_retry_index(idx)
                        op = None
                    if op:
                        if self._is_replay(header):
                            self._pir_mark_retrieved(idx)
                        elif op == expect_op and sess == session_id:
                            if reply_token and header.get("reply_token") != reply_token:
                                self._pir_mark_retrieved(idx)
                            else:
                                self._pir_mark_retrieved(idx)
                                return tensor
                        else:
                            self._pir_mark_retrieved(idx)
                now = time.time()
                if now - last_log > 5:
                    self.logger.info(
                        f"[wait] still waiting for {expect_op} session={session_id} elapsed={now - start_wait:.1f}s"
                    )
                    last_log = now
                if timeout_sec and (now - start_wait) > timeout_sec:
                    raise TimeoutError(f"Timeout waiting for {expect_op} for session {session_id}")
                continue
            buckets = self.board.poll_buckets()
            if not buckets:
                self._sleep_poll()
                continue
            if self.poll_shuffle and len(buckets) > 1:
                random.shuffle(buckets)
            for b in buckets:
                if b["bucket_id"] != bucket_id:
                    continue
                self.bytes_received += len(b["payload"])
                try:
                    op, sess, sender, tensor, header = decode_message(
                        b["payload"],
                        self._shared_key_hex,
                        rotation_seconds=self.key_rotation_seconds,
                        rotation_grace=self.key_rotation_grace,
                    )
                except Exception as e:
                    decode_failures += 1
                    if decode_failures <= 3:
                        self.logger.info(f"[decode-fail] wait bucket={bucket_id} err={e}")
                    continue
                if self._is_replay(header):
                    continue
                if op == expect_op and sess == session_id:
                    if reply_token and header.get("reply_token") != reply_token:
                        continue
                    return tensor
            self._sleep_poll()
            now = time.time()
            if now - last_log > 5:
                self.logger.info(
                    f"[wait] still waiting for {expect_op} session={session_id} bucket={bucket_id} elapsed={now - start_wait:.1f}s"
                )
                last_log = now
            if timeout_sec and (now - start_wait) > timeout_sec:
                raise TimeoutError(f"Timeout waiting for {expect_op} for session {session_id}")

    # ---------------- evaluation ----------------

    def evaluate(self) -> float:
        if not self._shared_key_hex:
            # Re-pair for evaluation if training already released the session
            self._request_pair()

        batch_size = self.batch_size
        n = len(self.x_test)
        all_logits = []

        for i in range(0, n, batch_size):
            xb = tf.convert_to_tensor(self.x_test[i:i + batch_size], dtype=tf.float32)
            z_cut = self.M1(xb, training=False)
            z_cut_np = z_cut.numpy()

            session_id = f"{self.name}-eval-{i}-{uuid.uuid4().hex}"
            reply_token = uuid.uuid4().hex

            retries = 0
            while True:
                try:
                    if not self._shared_key_hex:
                        self._request_pair()

                    payload = encode_message(
                        "INFER_REQ",
                        session_id,
                        sender_pseudo=None,
                        tensor=z_cut_np,
                        key_str=self._shared_key_hex,
                        pad_multiple=self.pad_multiple,
                        reply_token=reply_token,
                        rotation_seconds=self.key_rotation_seconds,
                    )
                    self._sleep_send()
                    if self.use_pir:
                        bucket_id = ""
                        self._pir_post(payload)
                    else:
                        bucket_id = self.board.create_bucket(payload)
                    self.bytes_sent += len(payload)
                    self._maybe_send_dummy()
                    self.logger.info(
                        f"[eval] sent INFER_REQ session={session_id} bucket={bucket_id} batch={z_cut_np.shape[0]}"
                    )

                    z_mid_np = self._wait_for_response(
                        bucket_id,
                        session_id,
                        "INFER_RES",
                        reply_token=reply_token,
                        timeout_sec=60,
                    )
                    if not self.use_pir:
                        self.board.ack_bucket(bucket_id)
                    break
                except TimeoutError as e:
                    retries += 1
                    self.logger.error(
                        f"[eval] timeout waiting for INFER_RES session={session_id} attempt={retries}; re-pairing and retrying"
                    )
                    self._send_session_done()
                    if retries >= 3:
                        raise e
                    # clear key so _request_pair happens on next loop
                    self._shared_key_hex = None
                    time.sleep(0.5)

            z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)
            logits = self.M3(z_mid, training=False)
            all_logits.append(logits.numpy())

        logits_full = np.concatenate(all_logits, axis=0)[:n]
        acc = batch_accuracy(self.y_test, logits_full)
        self.logger.info(f"Test accuracy: {acc:.4f}")
        self._last_acc = float(acc)
        self._send_session_done()
        self.logger.info(
            f"[bytes-summary] sent={self.bytes_sent}B recv={self.bytes_received}B "
            f"fwd={self._fwd_count} bwd={self._bwd_count} infer={self._infer_count}"
        )
        self.logger.info(
            "Summary name=%s target_m2=paired bytes_sent=%d bytes_received=%d acc=%.4f",
            self.name,
            self.bytes_sent,
            self.bytes_received,
            self._last_acc,
        )
        return float(acc)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "sender_id": "",
            "target_m2": "paired",
            "pseudonym_scope": "double_blind",
            "hide_sender": True,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "last_acc": self._last_acc,
        }

    def _send_session_done(self):
        if not self._shared_key_hex:
            return
        session_id = f"{self.name}-session-done-{uuid.uuid4().hex}"
        payload = encode_message(
            "SESSION_DONE",
            session_id,
            sender_pseudo=None,
            tensor=np.array([], dtype=np.float32),
            key_str=self._shared_key_hex,
            session_done=True,
            pad_multiple=self.pad_multiple,
            rotation_seconds=self.key_rotation_seconds,
        )
        if self.use_pir:
            self._pir_post(payload)
        else:
            self.board.create_bucket(payload)
        # Clear keys so the next request uses a fresh DH exchange.
        self._reset_pair_state()
        try:
            self.pairing.poll_assignment_client(self.name)
        except Exception:
            pass
        self.bytes_sent += len(payload)


# ============================================================
# Main entrypoint
# ============================================================


def main(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_cfg = cfg.get("run", {})
    general = cfg.get("general", {})
    db_cfg = cfg.get("double_blind", {})

    base_dir = run_cfg.get("base_dir", "runs")
    run_name = run_cfg.get("name") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    epochs = int(general.get("epochs", 2))
    batch_size = int(general.get("batch_size", 128))
    lr = float(general.get("lr", 1e-3))
    max_steps_per_epoch = int(general.get("max_steps_per_epoch", 0))
    suppress_warnings = bool(general.get("suppress_warnings", False))
    log_level = general.get("log_level", "INFO")
    model_arch = general.get("model_architecture", "default")
    dataset = general.get("dataset", "mnist")
    m1_model = model_arch
    m2_model = model_arch
    m3_model = model_arch
    if "pad_multiple" not in general:
        raise ValueError("general.pad_multiple must be defined in config.yaml")
    pad_multiple = int(general["pad_multiple"])

    key_rotation_seconds = int(general.get("key_rotation_seconds", 0))
    key_rotation_grace = int(general.get("key_rotation_grace", 1))
    replay_protection = bool(general.get("replay_protection", True))
    replay_cache_size = int(general.get("replay_cache_size", 10000))
    replay_window_ms = int(general.get("replay_window_ms", 300000))
    replay_future_ms = int(general.get("replay_future_ms", 60000))
    send_jitter_ms = int(general.get("send_jitter_ms", 0))
    poll_jitter_ms = int(general.get("poll_jitter_ms", 0))
    dummy_rate = float(general.get("dummy_message_rate", 0.0))
    m2_max_messages_per_poll = int(general.get("m2_max_messages_per_poll", 0))
    m2_batch_delay_ms = int(general.get("m2_batch_delay_ms", 0))
    poll_shuffle = bool(general.get("poll_shuffle", False))
    use_pir = bool(general.get("use_pir", False))
    pir_chunk_size = int(general.get("pir_chunk_size", 32))
    pir_clue_limit = int(general.get("pir_clue_limit", 0))

    board_host = general.get("board_host", "localhost")
    board_port = int(general.get("board_port", 50051))
    pairing_host = general.get("pairing_host", "localhost")
    pairing_port = int(general.get("pairing_port", 50052))

    m1m3_count = int(db_cfg.get("m1m3_count", 0))
    m2_count = int(db_cfg.get("m2_count", 0))
    if m1m3_count <= 0 or m2_count <= 0:
        raise ValueError("double_blind.m1m3_count and double_blind.m2_count must be > 0")

    global_logger = setup_global_logger(run_dir, log_level)
    global_logger.info(f"Run dir: {run_dir}")
    global_logger.info(
        f"Double-blind config: m1m3_count={m1m3_count}, m2_count={m2_count}, epochs={epochs}, batch_size={batch_size}, lr={lr}, model={model_arch}, dataset={dataset}"
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
    x_shards, y_shards = stratified_split(x_train, y_train, m1m3_count, seed=seed)

    m2_peers = []
    for i in range(m2_count):
        name = f"db_m2_{i + 1}"
        actor = DoubleBlindPeerM2.remote(
            name=name,
            run_dir=run_dir,
            board_host=board_host,
            board_port=board_port,
            pairing_host=pairing_host,
            pairing_port=pairing_port,
            m1_model=m1_model,
            m2_model=m2_model,
            m3_model=m3_model,
            pad_multiple=pad_multiple,
            input_dim=128,
            lr=lr,
            log_level=log_level,
            verbose=bool(general.get("m2_verbose", False)),
            log_every=int(general.get("m2_log_every", 50)),
            key_rotation_seconds=key_rotation_seconds,
            key_rotation_grace=key_rotation_grace,
            replay_protection=replay_protection,
            replay_cache_size=replay_cache_size,
            replay_window_ms=replay_window_ms,
            replay_future_ms=replay_future_ms,
            send_jitter_ms=send_jitter_ms,
            poll_jitter_ms=poll_jitter_ms,
            dummy_rate=dummy_rate,
            max_messages_per_poll=m2_max_messages_per_poll,
            batch_delay_ms=m2_batch_delay_ms,
            poll_shuffle=poll_shuffle,
            use_pir=use_pir,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
        )
        m2_peers.append(actor)
        global_logger.info(f"Spawned double-blind M2 peer: {name}")

    for actor in m2_peers:
        actor.run.remote()

    clients = []
    for i in range(m1m3_count):
        name = f"db_client_{i + 1}"
        client = DoubleBlindPeerM1M3.remote(
            name=name,
            run_dir=run_dir,
            x_train=x_shards[i],
            y_train=y_shards[i],
            x_test=x_test,
            y_test=y_test,
            board_host=board_host,
            board_port=board_port,
            pairing_host=pairing_host,
            pairing_port=pairing_port,
            m1_model=m1_model,
            m2_model=m2_model,
            m3_model=m3_model,
            pad_multiple=pad_multiple,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_steps_per_epoch=max_steps_per_epoch,
            log_level=log_level,
            verbose=bool(general.get("m1m3_verbose", False)),
            log_every=int(general.get("m1m3_log_every", 50)),
            key_rotation_seconds=key_rotation_seconds,
            key_rotation_grace=key_rotation_grace,
            replay_protection=replay_protection,
            replay_cache_size=replay_cache_size,
            replay_window_ms=replay_window_ms,
            replay_future_ms=replay_future_ms,
            send_jitter_ms=send_jitter_ms,
            poll_jitter_ms=poll_jitter_ms,
            dummy_rate=dummy_rate,
            poll_shuffle=poll_shuffle,
            use_pir=use_pir,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
        )
        clients.append(client)
        global_logger.info(f"Spawned double-blind M1M3 peer: {name}")

    global_logger.info("Starting double-blind training for all clients...")
    ray.get([c.train.remote() for c in clients])

    global_logger.info("Evaluating clients on test set (double-blind mode)...")
    accs = ray.get([c.evaluate.remote() for c in clients])
    for i, acc in enumerate(accs):
        global_logger.info(f"Client db_client_{i + 1} final test accuracy: {acc:.4f}")

    summaries = ray.get([c.summary.remote() for c in clients])
    for summary in summaries:
        global_logger.info(
            "Client summary name=%s sender_id=%s target_m2=%s pseudonym_scope=%s hide_sender=%s bytes_sent=%d bytes_received=%d acc=%s",
            summary.get("name"),
            summary.get("sender_id"),
            summary.get("target_m2"),
            summary.get("pseudonym_scope"),
            summary.get("hide_sender"),
            summary.get("bytes_sent"),
            summary.get("bytes_received"),
            summary.get("last_acc"),
        )


if __name__ == "__main__":
    main()
