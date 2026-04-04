from __future__ import annotations

import logging
import random
import uuid
import time
from collections import deque
from typing import Callable, Tuple

import grpc
import numpy as np
import tensorflow as tf
from keras import optimizers

import board_pb2
import board_pb2_grpc
from clients.bucket_board_client import BucketBoardClient
from pir_utils import (
    DEFAULT_MARKER,
    chunks_to_bytes,
    clue_matches,
    enc_number_from_bytes,
    enc_number_to_bytes,
    encrypt_clue,
    public_key_to_bytes,
)
from perf_utils import PerfWriter


EncodeFn = Callable[..., bytes]
DecodeFn = Callable[[bytes], Tuple[str, str, str, np.ndarray, dict]]
DecodeWithHeaderFn = Callable[[bytes], Tuple[str, str, str, np.ndarray, dict]]


class ReplayProtector:
    def __init__(self, max_size: int = 10000):
        self.max_size = max(1, int(max_size))
        self._seen: set[str] = set()
        self._order: deque[str] = deque()

    def seen_or_add(self, msg_id: str) -> bool:
        if msg_id in self._seen:
            return True
        self._seen.add(msg_id)
        self._order.append(msg_id)
        while len(self._order) > self.max_size:
            old = self._order.popleft()
            self._seen.discard(old)
        return False


def stratified_split(x: np.ndarray, y: np.ndarray, n_shards: int, seed: int = 42):
    """
    Split (x, y) into n_shards with class-balanced distribution.
    Returns two lists: x_shards, y_shards.
    """
    rng = np.random.default_rng(seed)
    x_shards = [[] for _ in range(n_shards)]
    y_shards = [[] for _ in range(n_shards)]

    classes = np.unique(y)
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        perm = rng.permutation(cls_idx)
        for i, idx in enumerate(perm):
            shard = i % n_shards
            x_shards[shard].append(x[idx])
            y_shards[shard].append(y[idx])

    for i in range(n_shards):
        if x_shards[i]:
            x_shards[i] = np.stack(x_shards[i], axis=0)
            y_shards[i] = np.array(y_shards[i], dtype=y.dtype)
        else:
            x_shards[i] = np.empty((0,) + x.shape[1:], dtype=x.dtype)
            y_shards[i] = np.empty((0,), dtype=y.dtype)
    return x_shards, y_shards


class BaseBoardClient:
    """Thin gRPC wrapper matching the legacy Board API."""

    def __init__(self, host: str, port: int, logger: logging.Logger | None = None):
        self.host = host
        self.port = port
        self.logger = logger or logging.getLogger("BaseBoardClient")
        grpc_opts = [
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(f"{host}:{port}", options=grpc_opts)
        self.stub = board_pb2_grpc.BoardServiceStub(self.channel)
        self.logger.info("Connected Board client to %s:%d", host, port)

    def post_message(self, sender: str, receiver: str, payload: bytes) -> str:
        req = board_pb2.PostMessageRequest(sender=sender, receiver=receiver, payload=payload, audience="")
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


class BasePoolBoardClient:
    """gRPC wrapper for Board pool API (audience-based)."""

    def __init__(self, host: str, port: int, logger: logging.Logger | None = None):
        self.host = host
        self.port = port
        self.logger = logger or logging.getLogger("BasePoolBoardClient")
        grpc_opts = [
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(f"{host}:{port}", options=grpc_opts)
        self.stub = board_pb2_grpc.BoardServiceStub(self.channel)
        self.logger.info("Connected Pool Board client to %s:%d", host, port)

    def post_to_pool(self, sender: str, payload: bytes, audience: str, receiver: str = "") -> str:
        req = board_pb2.PostMessageRequest(
            sender=sender,
            receiver=receiver,
            payload=payload,
            audience=audience,
        )
        resp = self.stub.PostMessage(req)
        return resp.msg_id

    def poll_pool(self, audience: str):
        req = board_pb2.PollPoolRequest(audience=audience)
        resp = self.stub.PollPool(req)
        return [
            {
                "msg_id": msg.msg_id,
                "sender": msg.sender,
                "receiver": msg.receiver,
                "payload": bytes(msg.payload),
                "timestamp": msg.timestamp_ms / 1000.0,
            }
            for msg in resp.messages
        ]

    def ack_message(self, msg_id: str, audience: str) -> bool:
        req = board_pb2.AckMessageRequest(msg_id=msg_id, audience=audience)
        resp = self.stub.AckMessage(req)
        return resp.removed


class BaseBucketBoardClient(BucketBoardClient):
    """Alias for clarity when injecting into bucket-based peers."""
    pass


class BasePIRBoardClient:
    """gRPC wrapper for PIR Board API."""

    def __init__(self, host: str, port: int, logger: logging.Logger | None = None):
        self.host = host
        self.port = port
        self.logger = logger or logging.getLogger("BasePIRBoardClient")
        grpc_opts = [
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(f"{host}:{port}", options=grpc_opts)
        self.stub = board_pb2_grpc.PIRServiceStub(self.channel)
        self.logger.info("Connected PIR Board client to %s:%d", host, port)

    def post_pir_message(self, payload: bytes, clue: bytes) -> int:
        req = board_pb2.PIRPostRequest(payload=payload, clue=clue)
        resp = self.stub.PostPIRMessage(req)
        return resp.index

    def get_clues(self, start_index: int = 0, limit: int = 0):
        req = board_pb2.PIRCluesRequest(start_index=start_index, limit=limit)
        resp = self.stub.GetClues(req)
        return [
            {
                "index": c.index,
                "clue": bytes(c.clue),
                "payload_len": c.payload_len,
            }
            for c in resp.clues
        ], resp.latest_index, resp.total

    def pir_query(self, public_key: bytes, enc_selector: list[bytes], chunk_size: int):
        req = board_pb2.PIRQueryRequest(
            public_key=public_key,
            enc_selector=enc_selector,
            chunk_size=chunk_size,
        )
        resp = self.stub.PIRQuery(req)
        return {
            "enc_chunks": [bytes(b) for b in resp.enc_chunks],
            "record_len": resp.record_len,
            "total": resp.total,
        }


class BasePeer:
    def __init__(
        self,
        name: str,
        logger: logging.Logger,
        verbose: bool = False,
        log_every: int = 50,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        use_pir: bool = False,
        pir_client: BasePIRBoardClient | None = None,
        pir_key: str = "",
        pir_chunk_size: int = 32,
        pir_clue_limit: int = 0,
        pir_rotation_seconds: int = 0,
        pir_rotation_grace: int = 1,
    ):
        self.name = name
        self.logger = logger
        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self.bytes_sent = 0
        self.bytes_received = 0
        self.send_jitter_ms = max(0, int(send_jitter_ms))
        self.poll_jitter_ms = max(0, int(poll_jitter_ms))
        self.dummy_rate = max(0.0, min(1.0, float(dummy_rate)))
        self.replay_window_ms = max(0, int(replay_window_ms))
        self.replay_future_ms = max(0, int(replay_future_ms))
        self._replay = ReplayProtector(replay_cache_size) if replay_protection else None
        self.use_pir = bool(use_pir)
        self.pir_client = pir_client
        self.pir_key = pir_key or ""
        self.pir_chunk_size = max(1, int(pir_chunk_size))
        self.pir_clue_limit = max(0, int(pir_clue_limit))
        self.pir_rotation_seconds = max(0, int(pir_rotation_seconds))
        self.pir_rotation_grace = max(0, int(pir_rotation_grace))
        if self.use_pir and self.pir_client is None:
            raise ValueError("use_pir=True requires pir_client")
        self._pir_scan_start = 0
        self._pir_total = 0
        self._pir_matches: deque[int] = deque()
        self._pir_payload_lens: dict[int, int] = {}
        self._pir_seen_indices: set[int] = set()
        self._pir_retrieved: set[int] = set()
        self._pir_retry_counts: dict[int, int] = {}
        self._last_decode_perf: dict | None = None

    def _encode_with_perf(self, encode_fn: EncodeFn, *args, **kwargs) -> tuple[bytes, dict]:
        perf: dict = {}
        try:
            payload = encode_fn(*args, perf=perf, **kwargs)
        except TypeError:
            payload = encode_fn(*args, **kwargs)
        return payload, perf

    def _decode_with_perf(self, decode_fn: DecodeFn, blob: bytes):
        perf: dict = {}
        try:
            result = decode_fn(blob, perf=perf)
        except TypeError:
            result = decode_fn(blob)
        if perf and len(result) >= 5:
            header = result[4]
            if isinstance(header, dict):
                header["_perf"] = perf
        return result

    def _sleep_poll(self, base_sec: float = 0.01):
        if self.poll_jitter_ms > 0:
            time.sleep(base_sec + random.uniform(0.0, self.poll_jitter_ms) / 1000.0)
        else:
            time.sleep(base_sec)

    def _sleep_send(self):
        if self.send_jitter_ms > 0:
            time.sleep(random.uniform(0.0, self.send_jitter_ms) / 1000.0)

    def _is_replay(self, header: dict | None) -> bool:
        if not self._replay or not header:
            return False
        msg_id = header.get("msg_id")
        if not msg_id:
            return False
        ts_ms = header.get("ts_ms")
        if ts_ms is not None and self.replay_window_ms:
            now_ms = int(time.time() * 1000)
            if ts_ms < (now_ms - self.replay_window_ms):
                return True
            if ts_ms > (now_ms + self.replay_future_ms):
                return True
        return self._replay.seen_or_add(msg_id)

    def _set_pir_key(self, key_str: str):
        self.pir_key = key_str or ""
        self._pir_scan_start = 0
        self._pir_total = 0
        self._pir_matches.clear()
        self._pir_payload_lens.clear()
        self._pir_seen_indices.clear()
        self._pir_retrieved.clear()
        self._pir_retry_counts.clear()

    def _pir_post(self, payload: bytes) -> int:
        clue = encrypt_clue(DEFAULT_MARKER, self.pir_key, rotation_seconds=self.pir_rotation_seconds)
        return self.pir_client.post_pir_message(payload, clue)

    def _pir_scan_clues(self):
        clues, latest, total = self.pir_client.get_clues(self._pir_scan_start, self.pir_clue_limit)
        if total is not None:
            self._pir_total = int(total)
        if not clues:
            return
        for clue in clues:
            idx = int(clue["index"])
            if idx in self._pir_seen_indices:
                continue
            self._pir_seen_indices.add(idx)
            if clue_matches(
                clue["clue"],
                self.pir_key,
                rotation_seconds=self.pir_rotation_seconds,
                rotation_grace=self.pir_rotation_grace,
                marker=DEFAULT_MARKER,
            ):
                self._pir_matches.append(idx)
                self._pir_payload_lens[idx] = int(clue["payload_len"])
        if self.pir_clue_limit > 0:
            self._pir_scan_start += len(clues)
        else:
            self._pir_scan_start = self._pir_total

    def _pir_query_index(self, index: int) -> bytes:
        try:
            from phe import paillier
        except Exception as exc:
            raise RuntimeError(f"phe not installed: {exc}") from exc

        public_key, private_key = paillier.generate_paillier_keypair(n_length=2048)
        max_bytes = max(1, (public_key.n.bit_length() - 1) // 8)
        chunk_size = min(self.pir_chunk_size, max_bytes)
        total = max(self._pir_total, index + 1)
        selector = [public_key.encrypt(1 if i == index else 0) for i in range(total)]
        enc_selector = [enc_number_to_bytes(x) for x in selector]
        resp = self.pir_client.pir_query(public_key_to_bytes(public_key), enc_selector, chunk_size)
        enc_chunks = [enc_number_from_bytes(public_key, b) for b in resp["enc_chunks"]]
        values = [private_key.decrypt(x) for x in enc_chunks]

        record_len = int(resp["record_len"])
        payload = chunks_to_bytes(values, chunk_size, record_len)
        payload_len = self._pir_payload_lens.get(index, record_len)
        return payload[:payload_len]

    def _pir_next_payload(self) -> tuple[int, bytes] | None:
        while True:
            if not self._pir_matches:
                self._pir_scan_clues()
                if not self._pir_matches:
                    return None
            idx = self._pir_matches.popleft()
            if idx in self._pir_retrieved:
                continue
            if self._pir_total <= idx:
                self._pir_scan_start = 0
                self._pir_matches.clear()
                continue
            payload = self._pir_query_index(idx)
            return idx, payload

    def _pir_mark_retrieved(self, idx: int):
        self._pir_retrieved.add(idx)
        if idx in self._pir_retry_counts:
            del self._pir_retry_counts[idx]

    def _pir_retry_index(self, idx: int, max_retries: int = 3):
        count = self._pir_retry_counts.get(idx, 0) + 1
        if count <= max_retries:
            self._pir_retry_counts[idx] = count
            try:
                payload_len = self._pir_payload_lens.get(idx)
                self.logger.info(
                    "[pir] retry idx=%d attempt=%d payload_len=%s",
                    idx,
                    count,
                    payload_len if payload_len is not None else "unknown",
                )
            except Exception:
                pass
            self._pir_matches.append(idx)
            return
        self._pir_retrieved.add(idx)
        if idx in self._pir_retry_counts:
            del self._pir_retry_counts[idx]


class BaseM2Peer(BasePeer):
    """Shared logic for peers holding M2 in direct Board mode (per-receiver queue)."""

    def __init__(
        self,
        name: str,
        board_client: BaseBoardClient,
        model,
        optimizer: optimizers.Optimizer,
        encode_fn: EncodeFn,
        decode_fn: DecodeFn,
        logger: logging.Logger,
        verbose: bool = False,
        log_every: int = 50,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        max_messages_per_poll: int = 1,
        batch_delay_ms: int = 0,
        use_pir: bool = False,
        pir_client: BasePIRBoardClient | None = None,
        pir_key: str = "",
        pir_chunk_size: int = 32,
        pir_clue_limit: int = 0,
        pir_rotation_seconds: int = 0,
        pir_rotation_grace: int = 1,
        perf_path: str | None = None,
    ):
        super().__init__(
            name=name,
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
            use_pir=use_pir,
            pir_client=pir_client,
            pir_key=pir_key,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
            pir_rotation_seconds=pir_rotation_seconds,
            pir_rotation_grace=pir_rotation_grace,
        )
        self.board_client = board_client
        self.M2 = model
        self.opt_M2 = optimizer
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self._sessions: dict[str, tuple] = {}
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0
        self._dummy_count = 0
        self._perf_writer = None
        if perf_path:
            fields = [
                "ts_ms",
                "peer",
                "op",
                "compute_ms",
                "encode_ms",
                "encode_serialize_ms",
                "encode_encrypt_ms",
                "decode_ms",
                "decode_decrypt_ms",
                "decode_deserialize_ms",
                "send_payload_bytes",
                "send_header_bytes",
                "send_pad_bytes",
                "send_crypto_bytes",
                "send_total_bytes",
                "recv_payload_bytes",
                "recv_header_bytes",
                "recv_pad_bytes",
                "recv_crypto_bytes",
                "recv_total_bytes",
            ]
            self._perf_writer = PerfWriter(perf_path, fields)
        self.max_messages_per_poll = max(1, int(max_messages_per_poll))
        self.batch_delay_ms = max(0, int(batch_delay_ms))
        self._dummy_count = 0

    def run(self):
        """Process messages addressed to this peer via the Board."""
        self.logger.info("M2 peer '%s' run() loop started.", self.name)
        while True:
            if self.use_pir:
                item = self._pir_next_payload()
                if not item:
                    self._sleep_poll()
                    continue
                idx, payload = item
                self.bytes_received += len(payload)
                try:
                    op, session, sender, tensor, header = self._decode_with_perf(self.decode_fn, payload)
                except Exception as exc:
                    self.logger.error("Failed to decode PIR message: %s", exc)
                    self._pir_retry_index(idx)
                    continue
                decode_perf = header.get("_perf") if isinstance(header, dict) else None

                if self._is_replay(header):
                    self._pir_mark_retrieved(idx)
                    continue

                if op == "DUMMY":
                    self._dummy_count += 1
                    self._pir_mark_retrieved(idx)
                    continue
                if op.endswith("_RES") or op == "SESSION_DONE":
                    self._pir_mark_retrieved(idx)
                    continue
                reply_token = header.get("reply_token")
                if op == "FWD_REQ":
                    self._handle_forward(session, sender, tensor, reply_token=reply_token, decode_perf=decode_perf)
                elif op == "BWD_REQ":
                    self._handle_backward(session, sender, tensor, reply_token=reply_token, decode_perf=decode_perf)
                elif op == "INFER_REQ":
                    self._handle_infer(session, sender, tensor, reply_token=reply_token, decode_perf=decode_perf)
                else:
                    self.logger.warning("Unknown op '%s' in session=%s from=%s", op, session, sender)
                self._pir_mark_retrieved(idx)
                total = self._fwd_count + self._bwd_count + self._infer_count
                if total and (total % self.log_every == 0):
                    self.logger.info(
                        "[bytes] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d dummy=%d",
                        self.bytes_sent,
                        self.bytes_received,
                        self._fwd_count,
                        self._bwd_count,
                        self._infer_count,
                        self._dummy_count,
                    )
                continue
            processed = 0
            while processed < self.max_messages_per_poll:
                msg = self.board_client.poll_message(receiver=self.name)
                if not msg:
                    break
                processed += 1
                self.bytes_received += len(msg["payload"])
                try:
                    op, session, sender, tensor, header = self._decode_with_perf(self.decode_fn, msg["payload"])
                except Exception as exc:  # best-effort robustness
                    self.logger.error("Failed to decode message: %s", exc)
                    continue
                decode_perf = header.get("_perf") if isinstance(header, dict) else None

                if self._is_replay(header):
                    continue

                if op == "DUMMY":
                    self._dummy_count += 1
                    continue
                if op.endswith("_RES") or op == "SESSION_DONE":
                    continue
                reply_token = header.get("reply_token")
                if op == "FWD_REQ":
                    self._handle_forward(session, sender, tensor, reply_token=reply_token, decode_perf=decode_perf)
                elif op == "BWD_REQ":
                    self._handle_backward(session, sender, tensor, reply_token=reply_token, decode_perf=decode_perf)
                elif op == "INFER_REQ":
                    self._handle_infer(session, sender, tensor, reply_token=reply_token, decode_perf=decode_perf)
                else:
                    self.logger.warning("Unknown op '%s' in session=%s from=%s", op, session, sender)

            if processed == 0:
                self._sleep_poll()
            elif self.batch_delay_ms > 0:
                time.sleep(self.batch_delay_ms / 1000.0)

            total = self._fwd_count + self._bwd_count + self._infer_count
            if total and (total % self.log_every == 0):
                self.logger.info(
                    "[bytes] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d dummy=%d",
                    self.bytes_sent,
                    self.bytes_received,
                    self._fwd_count,
                    self._bwd_count,
                    self._infer_count,
                    self._dummy_count,
                )

    def _handle_forward(
        self,
        session_id: str,
        sender: str,
        z_cut_np: np.ndarray,
        reply_token: str | None = None,
        decode_perf: dict | None = None,
    ):
        self._fwd_count += 1
        t0 = time.perf_counter()
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z_cut)
            z_mid = self.M2(z_cut, training=True)
        self._sessions[session_id] = (tape, z_cut, z_mid)
        z_mid_np = z_mid.numpy()
        compute_ms = (time.perf_counter() - t0) * 1000.0

        payload, encode_perf = self._encode_with_perf(
            self.encode_fn,
            "FWD_RES",
            session_id,
            self.name,
            z_mid_np,
            reply_token=reply_token,
        )
        self._sleep_send()
        if self.use_pir:
            self._pir_post(payload)
        else:
            self.board_client.post_message(sender=self.name, receiver=sender, payload=payload)
        self.bytes_sent += len(payload)
        self._record_m2_perf("FWD_REQ", decode_perf, encode_perf, compute_ms)

        if self.verbose and (self._fwd_count % self.log_every == 0):
            bs = z_cut_np.shape[0]
            feat = z_cut_np.shape[1] if z_cut_np.ndim > 1 else 1
            self.logger.info(
                "[FWD #%d] session=%s from=%s batch=%d feat_dim=%d",
                self._fwd_count,
                session_id,
                sender,
                bs,
                feat,
            )

    def _handle_backward(
        self,
        session_id: str,
        sender: str,
        dL_dz_mid_np: np.ndarray,
        reply_token: str | None = None,
        decode_perf: dict | None = None,
    ):
        self._bwd_count += 1
        if session_id not in self._sessions:
            self.logger.error("[BWD] no cached forward for session=%s", session_id)
            return

        tape, z_cut, z_mid = self._sessions.pop(session_id)
        t0 = time.perf_counter()
        dL_dz_mid = tf.convert_to_tensor(dL_dz_mid_np, dtype=tf.float32)
        targets = self.M2.trainable_variables + [z_cut]
        grads_all = tape.gradient(z_mid, targets, output_gradients=dL_dz_mid)
        grads_M2 = grads_all[:-1]
        dL_dz_cut = grads_all[-1]

        self.opt_M2.apply_gradients(zip(grads_M2, self.M2.trainable_variables))
        dL_dz_cut_np = dL_dz_cut.numpy()
        compute_ms = (time.perf_counter() - t0) * 1000.0

        payload, encode_perf = self._encode_with_perf(
            self.encode_fn,
            "BWD_RES",
            session_id,
            self.name,
            dL_dz_cut_np,
            reply_token=reply_token,
        )
        self._sleep_send()
        if self.use_pir:
            self._pir_post(payload)
        else:
            self.board_client.post_message(sender=self.name, receiver=sender, payload=payload)
        self.bytes_sent += len(payload)
        self._record_m2_perf("BWD_REQ", decode_perf, encode_perf, compute_ms)

        if self.verbose and (self._bwd_count % self.log_every == 0):
            grad_norm = tf.linalg.global_norm(grads_M2).numpy()
            dzcut_norm = tf.linalg.global_norm([dL_dz_cut]).numpy()
            self.logger.info(
                "[BWD #%d] session=%s from=%s grad_norm(M2)=%.4f grad_norm(dL/dz_cut)=%.4f",
                self._bwd_count,
                session_id,
                sender,
                grad_norm,
                dzcut_norm,
            )

    def _handle_infer(
        self,
        session_id: str,
        sender: str,
        z_cut_np: np.ndarray,
        reply_token: str | None = None,
        decode_perf: dict | None = None,
    ):
        self._infer_count += 1
        t0 = time.perf_counter()
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        z_mid = self.M2(z_cut, training=False)
        z_mid_np = z_mid.numpy()
        compute_ms = (time.perf_counter() - t0) * 1000.0

        payload, encode_perf = self._encode_with_perf(
            self.encode_fn,
            "INFER_RES",
            session_id,
            self.name,
            z_mid_np,
            reply_token=reply_token,
        )
        self._sleep_send()
        if self.use_pir:
            self._pir_post(payload)
        else:
            self.board_client.post_message(sender=self.name, receiver=sender, payload=payload)
        self.bytes_sent += len(payload)
        self._record_m2_perf("INFER_REQ", decode_perf, encode_perf, compute_ms)

        if self.verbose and (self._infer_count % self.log_every == 0):
            bs = z_cut_np.shape[0]
            self.logger.info(
                "[INFER #%d] session=%s from=%s batch=%d", self._infer_count, session_id, sender, bs
            )

    def _record_m2_perf(
        self,
        op: str,
        decode_perf: dict | None,
        encode_perf: dict | None,
        compute_ms: float,
    ):
        if not self._perf_writer:
            return
        row = {
            "ts_ms": int(time.time() * 1000),
            "peer": self.name,
            "op": op,
            "compute_ms": round(float(compute_ms), 4),
            "encode_ms": round(float((encode_perf or {}).get("total_ms", 0.0)), 4),
            "encode_serialize_ms": round(float((encode_perf or {}).get("serialize_ms", 0.0)), 4),
            "encode_encrypt_ms": round(float((encode_perf or {}).get("encrypt_ms", 0.0)), 4),
            "decode_ms": round(float((decode_perf or {}).get("total_ms", 0.0)), 4),
            "decode_decrypt_ms": round(float((decode_perf or {}).get("decrypt_ms", 0.0)), 4),
            "decode_deserialize_ms": round(float((decode_perf or {}).get("deserialize_ms", 0.0)), 4),
            "send_payload_bytes": int((encode_perf or {}).get("tensor_bytes", 0)),
            "send_header_bytes": int((encode_perf or {}).get("header_bytes", 0)),
            "send_pad_bytes": int((encode_perf or {}).get("pad_bytes", 0)),
            "send_crypto_bytes": int((encode_perf or {}).get("crypto_overhead_bytes", 0)),
            "send_total_bytes": int((encode_perf or {}).get("total_bytes", 0)),
            "recv_payload_bytes": int((decode_perf or {}).get("tensor_bytes", 0)),
            "recv_header_bytes": int((decode_perf or {}).get("header_bytes", 0)),
            "recv_pad_bytes": int((decode_perf or {}).get("pad_bytes", 0)),
            "recv_crypto_bytes": int((decode_perf or {}).get("crypto_overhead_bytes", 0)),
            "recv_total_bytes": int((decode_perf or {}).get("total_bytes", 0)),
        }
        self._perf_writer.write(row)


class BaseM1M3Peer(BasePeer):
    """Shared training/eval loop for M1+M3 peers in direct Board mode."""

    def __init__(
        self,
        name: str,
        run_dir: str,
        board_client: BaseBoardClient,
        M1,
        M3,
        opt_M1: optimizers.Optimizer,
        opt_M3: optimizers.Optimizer,
        loss_fn,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        target_m2: str,
        encode_fn: EncodeFn,
        decode_fn: DecodeFn,
        sender_id: str,
        batch_size: int,
        epochs: int,
        lr: float,
        logger: logging.Logger,
        max_steps_per_epoch: int = 0,
        verbose: bool = False,
        log_every: int = 50,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        use_pir: bool = False,
        pir_client: BasePIRBoardClient | None = None,
        pir_key: str = "",
        pir_chunk_size: int = 32,
        pir_clue_limit: int = 0,
        pir_rotation_seconds: int = 0,
        pir_rotation_grace: int = 1,
        perf_path: str | None = None,
    ):
        super().__init__(
            name=name,
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
            use_pir=use_pir,
            pir_client=pir_client,
            pir_key=pir_key,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
            pir_rotation_seconds=pir_rotation_seconds,
            pir_rotation_grace=pir_rotation_grace,
        )
        self.run_dir = run_dir
        self.board_client = board_client
        self.M1 = M1
        self.M3 = M3
        self.opt_M1 = opt_M1
        self.opt_M3 = opt_M3
        self.loss_fn = loss_fn
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.target_m2 = target_m2
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.sender_id = sender_id
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_steps_per_epoch = max(0, int(max_steps_per_epoch))
        self._seen_msg_ids: set[str] = set()
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0
        self._pending: dict[tuple[str, str], np.ndarray] = {}

        self.metrics_path = None  # set by subclass when ready
        self._perf_writer = None
        if perf_path:
            fields = [
                "peer",
                "epoch",
                "step",
                "step_ms",
                "m1_fwd_ms",
                "m1_bwd_ms",
                "m3_fwd_ms",
                "m3_bwd_ms",
                "encode_ms",
                "encode_serialize_ms",
                "encode_encrypt_ms",
                "decode_ms",
                "decode_decrypt_ms",
                "decode_deserialize_ms",
                "wait_ms",
                "send_payload_bytes",
                "send_header_bytes",
                "send_pad_bytes",
                "send_crypto_bytes",
                "send_total_bytes",
                "recv_payload_bytes",
                "recv_header_bytes",
                "recv_pad_bytes",
                "recv_crypto_bytes",
                "recv_total_bytes",
            ]
            self._perf_writer = PerfWriter(perf_path, fields)

    def _sender_for_session(self, session_id: str) -> str:
        return self.sender_id

    # The train/evaluate loops remain identical between vanilla and encrypted modes.
    def train_batches(self, write_metrics: Callable[[int, int, float, float], None]):
        n = len(self.x_train)
        steps_per_epoch = n // self.batch_size
        if self.max_steps_per_epoch and self.max_steps_per_epoch < steps_per_epoch:
            steps_per_epoch = self.max_steps_per_epoch
        self.logger.info("Starting training on %d samples, %d steps/epoch.", n, steps_per_epoch)

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
                step_start = time.perf_counter()
                step_perf = {
                    "peer": self.name,
                    "epoch": epoch,
                    "step": step,
                    "step_ms": 0.0,
                    "m1_fwd_ms": 0.0,
                    "m1_bwd_ms": 0.0,
                    "m3_fwd_ms": 0.0,
                    "m3_bwd_ms": 0.0,
                    "encode_ms": 0.0,
                    "encode_serialize_ms": 0.0,
                    "encode_encrypt_ms": 0.0,
                    "decode_ms": 0.0,
                    "decode_decrypt_ms": 0.0,
                    "decode_deserialize_ms": 0.0,
                    "wait_ms": 0.0,
                    "send_payload_bytes": 0,
                    "send_header_bytes": 0,
                    "send_pad_bytes": 0,
                    "send_crypto_bytes": 0,
                    "send_total_bytes": 0,
                    "recv_payload_bytes": 0,
                    "recv_header_bytes": 0,
                    "recv_pad_bytes": 0,
                    "recv_crypto_bytes": 0,
                    "recv_total_bytes": 0,
                }
                lo = step * self.batch_size
                hi = lo + self.batch_size
                xb = tf.convert_to_tensor(x_sh[lo:hi], dtype=tf.float32)
                yb = tf.convert_to_tensor(y_sh[lo:hi], dtype=tf.int32)

                t0 = time.perf_counter()
                with tf.GradientTape(persistent=True) as tape_M1:
                    z_cut = self.M1(xb, training=True)
                step_perf["m1_fwd_ms"] = (time.perf_counter() - t0) * 1000.0
                z_cut_np = z_cut.numpy()
                session_id = f"{self.name}-train-{epoch}-{step}-{uuid.uuid4().hex}"

                # Send forward request
                self._fwd_count += 1
                self._sleep_send()
                sender_pseudo = self._sender_for_session(session_id)
                payload, enc_perf = self._encode_with_perf(
                    self.encode_fn,
                    "FWD_REQ",
                    session_id,
                    sender_pseudo,
                    z_cut_np,
                )
                self._accum_encode_perf(step_perf, enc_perf)
                if self.use_pir:
                    self._pir_post(payload)
                else:
                    self.board_client.post_message(sender=self.sender_id, receiver=self.target_m2, payload=payload)
                self.bytes_sent += len(payload)
                self._maybe_send_dummy()
                if self.verbose and (self._fwd_count % self.log_every == 0):
                    self.logger.info(
                        "[FWD_REQ #%d] session=%s to=%s batch=%d",
                        self._fwd_count,
                        session_id,
                        self.target_m2,
                        z_cut_np.shape[0],
                    )

                wait_start = time.perf_counter()
                z_mid_np = self._wait_for(session_id, expect_op="FWD_RES")
                wait_ms = (time.perf_counter() - wait_start) * 1000.0
                last_decode = self._last_decode_perf or {}
                self._accum_decode_perf(step_perf, last_decode)
                step_perf["wait_ms"] += max(0.0, wait_ms - float(last_decode.get("total_ms", 0.0)))
                z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)

                t0 = time.perf_counter()
                with tf.GradientTape() as tape_M3:
                    tape_M3.watch(z_mid)
                    logits = self.M3(z_mid, training=True)
                    loss_value = self.loss_fn(yb, logits)
                step_perf["m3_fwd_ms"] = (time.perf_counter() - t0) * 1000.0

                targets = self.M3.trainable_variables + [z_mid]
                t0 = time.perf_counter()
                grads_all = tape_M3.gradient(loss_value, targets)
                grads_M3 = grads_all[:-1]
                dL_dz_mid = grads_all[-1]
                self.opt_M3.apply_gradients(zip(grads_M3, self.M3.trainable_variables))
                step_perf["m3_bwd_ms"] = (time.perf_counter() - t0) * 1000.0

                self._bwd_count += 1
                dL_dz_mid_np = dL_dz_mid.numpy()
                self._sleep_send()
                sender_pseudo = self._sender_for_session(session_id)
                payload, enc_perf = self._encode_with_perf(
                    self.encode_fn,
                    "BWD_REQ",
                    session_id,
                    sender_pseudo,
                    dL_dz_mid_np,
                )
                self._accum_encode_perf(step_perf, enc_perf)
                if self.use_pir:
                    self._pir_post(payload)
                else:
                    self.board_client.post_message(sender=self.sender_id, receiver=self.target_m2, payload=payload)
                self.bytes_sent += len(payload)
                self._maybe_send_dummy()
                if self.verbose and (self._bwd_count % self.log_every == 0):
                    self.logger.info("[BWD_REQ #%d] session=%s to=%s", self._bwd_count, session_id, self.target_m2)

                wait_start = time.perf_counter()
                dL_dz_cut_np = self._wait_for(session_id, expect_op="BWD_RES")
                wait_ms = (time.perf_counter() - wait_start) * 1000.0
                last_decode = self._last_decode_perf or {}
                self._accum_decode_perf(step_perf, last_decode)
                step_perf["wait_ms"] += max(0.0, wait_ms - float(last_decode.get("total_ms", 0.0)))
                dL_dz_cut = tf.convert_to_tensor(dL_dz_cut_np, dtype=tf.float32)

                t0 = time.perf_counter()
                grads_M1 = tape_M1.gradient(z_cut, self.M1.trainable_variables, output_gradients=dL_dz_cut)
                self.opt_M1.apply_gradients(zip(grads_M1, self.M1.trainable_variables))
                del tape_M1
                step_perf["m1_bwd_ms"] = (time.perf_counter() - t0) * 1000.0

                acc_batch = float(np.mean(np.argmax(logits.numpy(), axis=1) == yb.numpy()))
                loss_val = float(loss_value.numpy())
                epoch_losses.append(loss_val)
                epoch_accs.append(acc_batch)
                write_metrics(epoch, step, loss_val, acc_batch)

                if step % 100 == 0:
                    self.logger.info(
                        "Epoch %d Step %d/%d Loss=%.4f Acc=%.4f",
                        epoch,
                        step,
                        steps_per_epoch,
                        loss_val,
                        acc_batch,
                    )
                if self._perf_writer:
                    step_perf["step_ms"] = (time.perf_counter() - step_start) * 1000.0
                    self._perf_writer.write(self._round_perf(step_perf))

            mean_loss = float(np.mean(epoch_losses))
            mean_acc = float(np.mean(epoch_accs))
            elapsed = time.time() - start
            self.logger.info(
                "Epoch %d done in %.1fs -> Loss=%.4f Acc=%.4f", epoch, elapsed, mean_loss, mean_acc
            )
            self.logger.info(
                "[bytes-epoch] epoch=%d sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
                epoch,
                self.bytes_sent - sent_start,
                self.bytes_received - recv_start,
                self._fwd_count - fwd_start,
                self._bwd_count - bwd_start,
                self._infer_count - infer_start,
            )

        self.logger.info(
            "[bytes-summary] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
            self.bytes_sent,
            self.bytes_received,
            self._fwd_count,
            self._bwd_count,
            self._infer_count,
        )

    def evaluate_batches(self) -> float:
        batch_size = self.batch_size
        n = len(self.x_test)
        all_logits = []

        for i in range(0, n, batch_size):
            xb = tf.convert_to_tensor(self.x_test[i:i + batch_size], dtype=tf.float32)
            z_cut = self.M1(xb, training=False)
            z_cut_np = z_cut.numpy()
            session_id = f"{self.name}-eval-{i}-{uuid.uuid4().hex}"

            self._infer_count += 1
            self._sleep_send()
            sender_pseudo = self._sender_for_session(session_id)
            payload = self.encode_fn("INFER_REQ", session_id, sender_pseudo, z_cut_np)
            if self.use_pir:
                self._pir_post(payload)
            else:
                self.board_client.post_message(sender=self.sender_id, receiver=self.target_m2, payload=payload)
            self.bytes_sent += len(payload)
            self._maybe_send_dummy()
            if self.verbose and (self._infer_count % self.log_every == 0):
                self.logger.info(
                    "[INFER_REQ #%d] session=%s to=%s batch=%d",
                    self._infer_count,
                    session_id,
                    self.target_m2,
                    z_cut_np.shape[0],
                )

            z_mid_np = self._wait_for(session_id, expect_op="INFER_RES")
            z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)
            logits = self.M3(z_mid, training=False)
            all_logits.append(logits.numpy())

        logits_full = np.concatenate(all_logits, axis=0)[:n]
        acc = float(np.mean(np.argmax(logits_full, axis=1) == self.y_test))
        self.logger.info(
            "[bytes-summary] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
            self.bytes_sent,
            self.bytes_received,
            self._fwd_count,
            self._bwd_count,
            self._infer_count,
        )
        return acc

    def _wait_for(self, session_id: str, expect_op: str, reply_token: str | None = None) -> np.ndarray:
        if self.use_pir:
            token = reply_token or session_id
            cached = self._pending.pop((token, expect_op), None)
            if cached is not None:
                return cached
            while True:
                item = self._pir_next_payload()
                if not item:
                    self._sleep_poll()
                    continue
                idx, payload = item
                self.bytes_received += len(payload)
                try:
                    op, sess, sender_name, tensor, header = self._decode_with_perf(self.decode_fn, payload)
                except Exception:
                    self._pir_retry_index(idx)
                    continue
                if self._is_replay(header):
                    self._pir_mark_retrieved(idx)
                    continue
                if op == "DUMMY":
                    self._pir_mark_retrieved(idx)
                    continue
                msg_token = header.get("reply_token") or sess
                if op == expect_op and msg_token == token:
                    self._last_decode_perf = header.get("_perf") if isinstance(header, dict) else None
                    self._pir_mark_retrieved(idx)
                    return tensor
                self._pending[(msg_token, op)] = tensor
                self._pir_mark_retrieved(idx)
            return None

        while True:
            msg = self.board_client.poll_message(receiver=self.sender_id)
            if not msg:
                self._sleep_poll()
                continue
            msg_id = msg["msg_id"]
            if msg_id in self._seen_msg_ids:
                continue
            self._seen_msg_ids.add(msg_id)
            self.bytes_received += len(msg["payload"])
            try:
                op, sess, sender_name, tensor, header = self._decode_with_perf(self.decode_fn, msg["payload"])
            except Exception:
                continue
            if self._is_replay(header):
                continue
            if op == expect_op and sess == session_id:
                self._last_decode_perf = header.get("_perf") if isinstance(header, dict) else None
                return tensor
            self._sleep_poll()

    def _accum_encode_perf(self, step_perf: dict, perf: dict | None):
        if not perf:
            return
        step_perf["encode_ms"] += float(perf.get("total_ms", 0.0))
        step_perf["encode_serialize_ms"] += float(perf.get("serialize_ms", 0.0))
        step_perf["encode_encrypt_ms"] += float(perf.get("encrypt_ms", 0.0))
        step_perf["send_payload_bytes"] += int(perf.get("tensor_bytes", 0))
        step_perf["send_header_bytes"] += int(perf.get("header_bytes", 0))
        step_perf["send_pad_bytes"] += int(perf.get("pad_bytes", 0))
        step_perf["send_crypto_bytes"] += int(perf.get("crypto_overhead_bytes", 0))
        step_perf["send_total_bytes"] += int(perf.get("total_bytes", 0))

    def _accum_decode_perf(self, step_perf: dict, perf: dict | None):
        if not perf:
            return
        step_perf["decode_ms"] += float(perf.get("total_ms", 0.0))
        step_perf["decode_decrypt_ms"] += float(perf.get("decrypt_ms", 0.0))
        step_perf["decode_deserialize_ms"] += float(perf.get("deserialize_ms", 0.0))
        step_perf["recv_payload_bytes"] += int(perf.get("tensor_bytes", 0))
        step_perf["recv_header_bytes"] += int(perf.get("header_bytes", 0))
        step_perf["recv_pad_bytes"] += int(perf.get("pad_bytes", 0))
        step_perf["recv_crypto_bytes"] += int(perf.get("crypto_overhead_bytes", 0))
        step_perf["recv_total_bytes"] += int(perf.get("total_bytes", 0))

    @staticmethod
    def _round_perf(step_perf: dict) -> dict:
        rounded = {}
        for key, value in step_perf.items():
            if isinstance(value, float):
                rounded[key] = round(value, 4)
            else:
                rounded[key] = value
        return rounded

    def _maybe_send_dummy(self):
        if self.dummy_rate <= 0.0:
            return
        if random.random() >= self.dummy_rate:
            return
        try:
            dummy_session = f"{self.name}-dummy-{uuid.uuid4().hex}"
            dummy_tensor = np.zeros((0,), dtype=np.float32)
            sender_pseudo = self._sender_for_session(dummy_session)
            reply_token = self._reply_token_for_session(dummy_session)
            payload = self.encode_fn("DUMMY", dummy_session, sender_pseudo, dummy_tensor, reply_token=reply_token)
            if self.use_pir:
                self._pir_post(payload)
            else:
                self.board_client.post_message(sender=self.sender_id, receiver=self.target_m2, payload=payload)
            self.bytes_sent += len(payload)
        except Exception:
            pass


# ============================================================
# Pool board variants
# ============================================================


class BasePoolM2Peer(BasePeer):
    """Shared logic for M2 peers using the audience-based pool API."""

    def __init__(
        self,
        name: str,
        board_client: BasePoolBoardClient,
        model,
        optimizer: optimizers.Optimizer,
        encode_fn: EncodeFn,
        decode_fn: DecodeWithHeaderFn,
        logger: logging.Logger,
        audience_in: str,
        audience_out: str,
        target_name: str | None = None,
        verbose: bool = False,
        log_every: int = 50,
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
        hide_sender: bool = False,
        use_pir: bool = False,
        pir_client: BasePIRBoardClient | None = None,
        pir_key: str = "",
        pir_chunk_size: int = 32,
        pir_clue_limit: int = 0,
        pir_rotation_seconds: int = 0,
        pir_rotation_grace: int = 1,
    ):
        super().__init__(
            name=name,
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
            use_pir=use_pir,
            pir_client=pir_client,
            pir_key=pir_key,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
            pir_rotation_seconds=pir_rotation_seconds,
            pir_rotation_grace=pir_rotation_grace,
        )
        self.board = board_client
        self.M2 = model
        self.opt_M2 = optimizer
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.audience_in = audience_in
        self.audience_out = audience_out
        self.target_name = target_name
        self._sessions: dict[str, tuple] = {}
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0
        self._dummy_count = 0
        self.max_messages_per_poll = max(0, int(max_messages_per_poll))
        self.batch_delay_ms = max(0, int(batch_delay_ms))
        self.poll_shuffle = bool(poll_shuffle)
        self.hide_sender = bool(hide_sender)

    def run(self):
        self.logger.info("Pool M2 '%s' run() loop started.", self.name)
        while True:
            if self.use_pir:
                item = self._pir_next_payload()
                if not item:
                    self._sleep_poll()
                    continue
                idx, payload = item
                self.bytes_received += len(payload)
                try:
                    op, session, sender, tensor, header = self.decode_fn(payload)
                except Exception:
                    self._pir_retry_index(idx)
                    continue
                target = header.get("target_m2")
                if self.target_name and target and target != self.target_name:
                    self._pir_mark_retrieved(idx)
                    continue
                if self._is_replay(header):
                    self._pir_mark_retrieved(idx)
                    continue
                if op == "DUMMY":
                    self._dummy_count += 1
                    self._pir_mark_retrieved(idx)
                    continue
                reply_token = header.get("reply_token")
                if op == "FWD_REQ":
                    self._handle_forward(session, sender, tensor, reply_token=reply_token)
                elif op == "BWD_REQ":
                    self._handle_backward(session, sender, tensor, reply_token=reply_token)
                elif op == "INFER_REQ":
                    self._handle_infer(session, sender, tensor, reply_token=reply_token)
                else:
                    self.logger.warning("Unknown op '%s' in session=%s from=%s", op, session, sender)
                self._pir_mark_retrieved(idx)
                total = self._fwd_count + self._bwd_count + self._infer_count
                if total and (total % self.log_every == 0):
                    self.logger.info(
                        "[bytes] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d dummy=%d",
                        self.bytes_sent,
                        self.bytes_received,
                        self._fwd_count,
                        self._bwd_count,
                        self._infer_count,
                        self._dummy_count,
                    )
                continue
            msgs = self.board.poll_pool(self.audience_in)
            if not msgs:
                self._sleep_poll()
                continue
            if self.poll_shuffle and len(msgs) > 1:
                random.shuffle(msgs)
            if self.max_messages_per_poll > 0:
                msgs = msgs[: self.max_messages_per_poll]
            for msg in msgs:
                msg_id = msg["msg_id"]
                self.bytes_received += len(msg["payload"])
                try:
                    op, session, sender, tensor, header = self.decode_fn(msg["payload"])
                except Exception:
                    continue

                target = header.get("target_m2")
                if self.target_name and target and target != self.target_name:
                    continue

                if self._is_replay(header):
                    self.board.ack_message(msg_id, self.audience_in)
                    continue

                if op == "DUMMY":
                    self._dummy_count += 1
                    self.board.ack_message(msg_id, self.audience_in)
                    continue

                reply_token = header.get("reply_token")
                if op == "FWD_REQ":
                    self._handle_forward(session, sender, tensor, reply_token=reply_token)
                elif op == "BWD_REQ":
                    self._handle_backward(session, sender, tensor, reply_token=reply_token)
                elif op == "INFER_REQ":
                    self._handle_infer(session, sender, tensor, reply_token=reply_token)
                else:
                    self.logger.warning("Unknown op '%s' in session=%s from=%s", op, session, sender)
                self.board.ack_message(msg_id, self.audience_in)

            total = self._fwd_count + self._bwd_count + self._infer_count
            if total and (total % self.log_every == 0):
                self.logger.info(
                    "[bytes] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d dummy=%d",
                    self.bytes_sent,
                    self.bytes_received,
                    self._fwd_count,
                    self._bwd_count,
                    self._infer_count,
                    self._dummy_count,
                )
            if self.batch_delay_ms > 0:
                time.sleep(self.batch_delay_ms / 1000.0)

    def _handle_forward(
        self,
        session_id: str,
        sender: str,
        z_cut_np: np.ndarray,
        reply_token: str | None = None,
    ):
        self._fwd_count += 1
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z_cut)
            z_mid = self.M2(z_cut, training=True)
        self._sessions[session_id] = (tape, z_cut, z_mid)
        z_mid_np = z_mid.numpy()

        sender_id = "" if self.hide_sender else self.name
        payload = self.encode_fn("FWD_RES", session_id, sender_id, z_mid_np, reply_token=reply_token)
        self._sleep_send()
        if self.use_pir:
            self._pir_post(payload)
        else:
            self.board.post_to_pool(sender=sender_id, receiver="", payload=payload, audience=self.audience_out)
        self.bytes_sent += len(payload)

    def _handle_backward(
        self,
        session_id: str,
        sender: str,
        dL_dz_mid_np: np.ndarray,
        reply_token: str | None = None,
    ):
        self._bwd_count += 1
        if session_id not in self._sessions:
            self.logger.error("[BWD] no cached forward for session=%s", session_id)
            return

        tape, z_cut, z_mid = self._sessions.pop(session_id)
        dL_dz_mid = tf.convert_to_tensor(dL_dz_mid_np, dtype=tf.float32)
        targets = self.M2.trainable_variables + [z_cut]
        grads_all = tape.gradient(z_mid, targets, output_gradients=dL_dz_mid)
        grads_M2 = grads_all[:-1]
        dL_dz_cut = grads_all[-1]

        self.opt_M2.apply_gradients(zip(grads_M2, self.M2.trainable_variables))
        dL_dz_cut_np = dL_dz_cut.numpy()

        sender_id = "" if self.hide_sender else self.name
        payload = self.encode_fn("BWD_RES", session_id, sender_id, dL_dz_cut_np, reply_token=reply_token)
        self._sleep_send()
        if self.use_pir:
            self._pir_post(payload)
        else:
            self.board.post_to_pool(sender=sender_id, receiver="", payload=payload, audience=self.audience_out)
        self.bytes_sent += len(payload)

    def _handle_infer(
        self,
        session_id: str,
        sender: str,
        z_cut_np: np.ndarray,
        reply_token: str | None = None,
    ):
        self._infer_count += 1
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        z_mid = self.M2(z_cut, training=False)
        z_mid_np = z_mid.numpy()

        sender_id = "" if self.hide_sender else self.name
        payload = self.encode_fn("INFER_RES", session_id, sender_id, z_mid_np, reply_token=reply_token)
        self._sleep_send()
        if self.use_pir:
            self._pir_post(payload)
        else:
            self.board.post_to_pool(sender=sender_id, receiver="", payload=payload, audience=self.audience_out)
        self.bytes_sent += len(payload)


class BasePoolM1M3Peer(BasePeer):
    """Shared training/eval loop for M1+M3 peers using pool Board API."""

    def __init__(
        self,
        name: str,
        run_dir: str,
        board_client: BasePoolBoardClient,
        M1,
        M3,
        opt_M1: optimizers.Optimizer,
        opt_M3: optimizers.Optimizer,
        loss_fn,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        target_m2: str,
        encode_fn: EncodeFn,
        decode_fn: DecodeWithHeaderFn,
        sender_id: str,
        audience_in: str,
        audience_out: str,
        batch_size: int,
        epochs: int,
        lr: float,
        logger: logging.Logger,
        max_steps_per_epoch: int = 0,
        verbose: bool = False,
        log_every: int = 50,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        pseudonym_scope: str = "per_run",
        poll_shuffle: bool = False,
        hide_sender: bool = False,
        use_pir: bool = False,
        pir_client: BasePIRBoardClient | None = None,
        pir_key: str = "",
        pir_chunk_size: int = 32,
        pir_clue_limit: int = 0,
        pir_rotation_seconds: int = 0,
        pir_rotation_grace: int = 1,
    ):
        super().__init__(
            name=name,
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
            use_pir=use_pir,
            pir_client=pir_client,
            pir_key=pir_key,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
            pir_rotation_seconds=pir_rotation_seconds,
            pir_rotation_grace=pir_rotation_grace,
        )
        self.run_dir = run_dir
        self.board = board_client
        self.M1 = M1
        self.M3 = M3
        self.opt_M1 = opt_M1
        self.opt_M3 = opt_M3
        self.loss_fn = loss_fn
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.target_m2 = target_m2
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.sender_id = sender_id
        self.audience_in = audience_in
        self.audience_out = audience_out
        self.pseudonym_scope = pseudonym_scope
        self.poll_shuffle = bool(poll_shuffle)
        self.hide_sender = bool(hide_sender)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_steps_per_epoch = max(0, int(max_steps_per_epoch))
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0
        self._pending: dict[tuple[str, str], np.ndarray] = {}
        self._session_pseudos: dict[str, str] = {}
        self._session_tokens: dict[str, str] = {}

        self.metrics_path = None  # set by subclass when ready

    def _wait_for(self, session_id: str, expect_op: str, reply_token: str | None = None) -> np.ndarray:
        token = reply_token or session_id
        cached = self._pending.pop((token, expect_op), None)
        if cached is not None:
            return cached
        while True:
            if self.use_pir:
                item = self._pir_next_payload()
                if not item:
                    self._sleep_poll()
                    continue
                idx, payload = item
                self.bytes_received += len(payload)
                try:
                    op, sess, sender_name, tensor, header = self.decode_fn(payload)
                except Exception:
                    self._pir_retry_index(idx)
                    continue
                if self._is_replay(header):
                    self._pir_mark_retrieved(idx)
                    continue
                if op == "DUMMY":
                    self._pir_mark_retrieved(idx)
                    continue
                msg_token = header.get("reply_token") or sess
                if op == expect_op and msg_token == token:
                    self._pir_mark_retrieved(idx)
                    return tensor
                self._pending[(msg_token, op)] = tensor
                self._pir_mark_retrieved(idx)
                self._sleep_poll()
                continue

            msgs = self.board.poll_pool(self.audience_out)
            if not msgs:
                self._sleep_poll()
                continue
            if self.poll_shuffle and len(msgs) > 1:
                random.shuffle(msgs)
            for msg in msgs:
                msg_id = msg["msg_id"]
                self.bytes_received += len(msg["payload"])
                try:
                    op, sess, sender_name, tensor, header = self.decode_fn(msg["payload"])
                except Exception:
                    continue
                if self._is_replay(header):
                    self.board.ack_message(msg_id, self.audience_out)
                    continue
                if op == "DUMMY":
                    self.board.ack_message(msg_id, self.audience_out)
                    continue
                msg_token = header.get("reply_token") or sess
                if op == expect_op and msg_token == token:
                    self.board.ack_message(msg_id, self.audience_out)
                    return tensor
                self._pending[(msg_token, op)] = tensor
                self.board.ack_message(msg_id, self.audience_out)
            self._sleep_poll()

    def _sender_for_session(self, session_id: str) -> str:
        if self.hide_sender:
            return ""
        if self.pseudonym_scope != "per_session":
            return self.sender_id
        pseudo = self._session_pseudos.get(session_id)
        if not pseudo:
            pseudo = f"cli_{uuid.uuid4().hex[:8]}"
            self._session_pseudos[session_id] = pseudo
        return pseudo

    def _reply_token_for_session(self, session_id: str) -> str:
        token = self._session_tokens.get(session_id)
        if not token:
            token = uuid.uuid4().hex
            self._session_tokens[session_id] = token
        return token

    def train_batches(self, write_metrics: Callable[[int, int, float, float], None]):
        n = len(self.x_train)
        steps_per_epoch = n // self.batch_size
        if self.max_steps_per_epoch and self.max_steps_per_epoch < steps_per_epoch:
            steps_per_epoch = self.max_steps_per_epoch
        self.logger.info("Starting training on %d samples, %d steps/epoch (pool).", n, steps_per_epoch)

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

                with tf.GradientTape(persistent=True) as tape_M1:
                    z_cut = self.M1(xb, training=True)
                z_cut_np = z_cut.numpy()
                session_id = f"{self.name}-train-{epoch}-{step}-{uuid.uuid4().hex}"

                self._fwd_count += 1
                self._sleep_send()
                sender_pseudo = self._sender_for_session(session_id)
                reply_token = self._reply_token_for_session(session_id)
                payload = self.encode_fn("FWD_REQ", session_id, sender_pseudo, z_cut_np, reply_token=reply_token)
                if self.use_pir:
                    self._pir_post(payload)
                else:
                    self.board.post_to_pool(
                        sender=sender_pseudo,
                        receiver="",
                        payload=payload,
                        audience=self.audience_in,
                    )
                self.bytes_sent += len(payload)
                self._maybe_send_dummy()

                z_mid_np = self._wait_for(session_id, expect_op="FWD_RES", reply_token=reply_token)
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
                self._sleep_send()
                payload = self.encode_fn("BWD_REQ", session_id, sender_pseudo, dL_dz_mid_np, reply_token=reply_token)
                if self.use_pir:
                    self._pir_post(payload)
                else:
                    self.board.post_to_pool(
                        sender=sender_pseudo,
                        receiver="",
                        payload=payload,
                        audience=self.audience_in,
                    )
                self.bytes_sent += len(payload)
                self._maybe_send_dummy()

                dL_dz_cut_np = self._wait_for(session_id, expect_op="BWD_RES", reply_token=reply_token)
                dL_dz_cut = tf.convert_to_tensor(dL_dz_cut_np, dtype=tf.float32)

                grads_M1 = tape_M1.gradient(z_cut, self.M1.trainable_variables, output_gradients=dL_dz_cut)
                self.opt_M1.apply_gradients(zip(grads_M1, self.M1.trainable_variables))
                del tape_M1

                acc_batch = float(np.mean(np.argmax(logits.numpy(), axis=1) == yb.numpy()))
                loss_val = float(loss_value.numpy())
                epoch_losses.append(loss_val)
                epoch_accs.append(acc_batch)
                write_metrics(epoch, step, loss_val, acc_batch)

                if step % 100 == 0:
                    self.logger.info(
                        "Epoch %d Step %d/%d Loss=%.4f Acc=%.4f",
                        epoch,
                        step,
                        steps_per_epoch,
                        loss_val,
                        acc_batch,
                    )

            mean_loss = float(np.mean(epoch_losses))
            mean_acc = float(np.mean(epoch_accs))
            elapsed = time.time() - start
            self.logger.info(
                "Epoch %d done in %.1fs -> Loss=%.4f Acc=%.4f", epoch, elapsed, mean_loss, mean_acc
            )
            self.logger.info(
                "[bytes-epoch] epoch=%d sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
                epoch,
                self.bytes_sent - sent_start,
                self.bytes_received - recv_start,
                self._fwd_count - fwd_start,
                self._bwd_count - bwd_start,
                self._infer_count - infer_start,
            )

        self.logger.info(
            "[bytes-summary] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
            self.bytes_sent,
            self.bytes_received,
            self._fwd_count,
            self._bwd_count,
            self._infer_count,
        )

    def evaluate_batches(self) -> float:
        batch_size = self.batch_size
        n = len(self.x_test)
        all_logits = []

        for i in range(0, n, batch_size):
            xb = tf.convert_to_tensor(self.x_test[i:i + batch_size], dtype=tf.float32)
            z_cut = self.M1(xb, training=False)
            z_cut_np = z_cut.numpy()
            session_id = f"{self.name}-eval-{i}-{uuid.uuid4().hex}"

            self._infer_count += 1
            self._sleep_send()
            sender_pseudo = self._sender_for_session(session_id)
            reply_token = self._reply_token_for_session(session_id)
            payload = self.encode_fn("INFER_REQ", session_id, sender_pseudo, z_cut_np, reply_token=reply_token)
            if self.use_pir:
                self._pir_post(payload)
            else:
                self.board.post_to_pool(
                    sender=sender_pseudo,
                    receiver="",
                    payload=payload,
                    audience=self.audience_in,
                )
            self.bytes_sent += len(payload)
            self._maybe_send_dummy()

            z_mid_np = self._wait_for(session_id, expect_op="INFER_RES", reply_token=reply_token)
            z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)
            logits = self.M3(z_mid, training=False)
            all_logits.append(logits.numpy())

        logits_full = np.concatenate(all_logits, axis=0)[:n]
        acc = float(np.mean(np.argmax(logits_full, axis=1) == self.y_test))
        self.logger.info(
            "[bytes-summary] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
            self.bytes_sent,
            self.bytes_received,
            self._fwd_count,
            self._bwd_count,
            self._infer_count,
        )
        return acc

    def _maybe_send_dummy(self):
        if self.dummy_rate <= 0.0:
            return
        if random.random() >= self.dummy_rate:
            return
        try:
            dummy_session = f"{self.name}-dummy-{uuid.uuid4().hex}"
            dummy_tensor = np.zeros((0,), dtype=np.float32)
            sender_pseudo = self._sender_for_session(dummy_session)
            reply_token = self._reply_token_for_session(dummy_session)
            payload = self.encode_fn("DUMMY", dummy_session, sender_pseudo, dummy_tensor, reply_token=reply_token)
            if self.use_pir:
                self._pir_post(payload)
            else:
                self.board.post_to_pool(
                    sender=sender_pseudo,
                    receiver="",
                    payload=payload,
                    audience=self.audience_in,
                )
            self.bytes_sent += len(payload)
        except Exception:
            pass


# ============================================================
# Bucket board variants
# ============================================================


class BaseBucketM2Peer(BasePeer):
    """Shared logic for M2 peers using the bucket Board API."""

    def __init__(
        self,
        name: str,
        board_client: BucketBoardClient,
        model,
        optimizer: optimizers.Optimizer,
        encode_fn: EncodeFn,
        decode_fn: DecodeWithHeaderFn,
        logger: logging.Logger,
        target_name: str | None = None,
        verbose: bool = False,
        log_every: int = 50,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        bucket_namespace: str = "",
        max_messages_per_poll: int = 0,
        batch_delay_ms: int = 0,
        poll_shuffle: bool = False,
        use_pir: bool = False,
        pir_client: BasePIRBoardClient | None = None,
        pir_key: str = "",
        pir_chunk_size: int = 32,
        pir_clue_limit: int = 0,
        pir_rotation_seconds: int = 0,
        pir_rotation_grace: int = 1,
    ):
        super().__init__(
            name=name,
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
            use_pir=use_pir,
            pir_client=pir_client,
            pir_key=pir_key,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
            pir_rotation_seconds=pir_rotation_seconds,
            pir_rotation_grace=pir_rotation_grace,
        )
        self.board = board_client
        self.M2 = model
        self.opt_M2 = optimizer
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.target_name = target_name
        self.bucket_namespace = bucket_namespace or ""
        self.max_messages_per_poll = max(0, int(max_messages_per_poll))
        self.batch_delay_ms = max(0, int(batch_delay_ms))
        self.poll_shuffle = bool(poll_shuffle)
        self.bucket_namespace = bucket_namespace or ""
        self._sessions: dict[str, tuple] = {}
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0

    def process_bucket(self, bucket: dict):
        bid = bucket["bucket_id"]
        self.bytes_received += len(bucket["payload"])
        try:
            op, session, sender, tensor, header = self.decode_fn(bucket["payload"])
        except Exception:
            return
        # Optional target filtering
        target = header.get("target_m2")
        if self.target_name and target and target != self.target_name:
            return
        if self._is_replay(header):
            self.board.ack_bucket(bid)
            return

        reply_token = header.get("reply_token")
        if op == "DUMMY":
            self._dummy_count += 1
            self.board.ack_bucket(bid)
            return
        if op == "FWD_REQ":
            self._handle_forward(bid, session, sender, tensor, reply_token=reply_token)
        elif op == "BWD_REQ":
            self._handle_backward(bid, session, sender, tensor, reply_token=reply_token)
        elif op == "INFER_REQ":
            self._handle_infer(bid, session, sender, tensor, reply_token=reply_token)

    def run(self):
        self.logger.info("Bucket M2 '%s' run() loop started.", self.name)
        while True:
            if self.use_pir:
                item = self._pir_next_payload()
                if not item:
                    self._sleep_poll()
                    continue
                idx, payload = item
                self.bytes_received += len(payload)
                try:
                    op, session, sender, tensor, header = self.decode_fn(payload)
                except Exception:
                    self._pir_retry_index(idx)
                    continue
                target = header.get("target_m2")
                if self.target_name and target and target != self.target_name:
                    self._pir_mark_retrieved(idx)
                    continue
                if self._is_replay(header):
                    self._pir_mark_retrieved(idx)
                    continue
                reply_token = header.get("reply_token")
                if op == "DUMMY":
                    self._dummy_count += 1
                    self._pir_mark_retrieved(idx)
                    continue
                if op.endswith("_RES") or op == "SESSION_DONE":
                    self._pir_mark_retrieved(idx)
                    continue
                if op == "FWD_REQ":
                    self._handle_forward("", session, sender, tensor, reply_token=reply_token)
                elif op == "BWD_REQ":
                    self._handle_backward("", session, sender, tensor, reply_token=reply_token)
                elif op == "INFER_REQ":
                    self._handle_infer("", session, sender, tensor, reply_token=reply_token)
                self._pir_mark_retrieved(idx)
                total = self._fwd_count + self._bwd_count + self._infer_count
                if total and (total % self.log_every == 0):
                    self.logger.info(
                        "[bytes] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d dummy=%d",
                        self.bytes_sent,
                        self.bytes_received,
                        self._fwd_count,
                        self._bwd_count,
                        self._infer_count,
                        self._dummy_count,
                    )
                continue
            buckets = self.board.poll_buckets(namespace=self.bucket_namespace)
            if not buckets:
                self._sleep_poll()
                continue
            if self.poll_shuffle and len(buckets) > 1:
                random.shuffle(buckets)
            if self.max_messages_per_poll > 0:
                buckets = buckets[: self.max_messages_per_poll]
            for b in buckets:
                self.process_bucket(b)
            total = self._fwd_count + self._bwd_count + self._infer_count
            if total and (total % self.log_every == 0):
                self.logger.info(
                    "[bytes] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d dummy=%d",
                    self.bytes_sent,
                    self.bytes_received,
                    self._fwd_count,
                    self._bwd_count,
                    self._infer_count,
                    self._dummy_count,
                )
            if self.batch_delay_ms > 0:
                time.sleep(self.batch_delay_ms / 1000.0)

    # ----- handlers -----
    def _handle_forward(
        self,
        bucket_id: str,
        session_id: str,
        sender: str,
        z_cut_np: np.ndarray,
        reply_token: str | None = None,
    ):
        self._fwd_count += 1
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z_cut)
            z_mid = self.M2(z_cut, training=True)
        self._sessions[session_id] = (tape, z_cut, z_mid)
        z_mid_np = z_mid.numpy()

        resp = self.encode_fn("FWD_RES", session_id, self.name, z_mid_np, reply_token=reply_token)
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
        self._bwd_count += 1
        if session_id not in self._sessions:
            self.logger.error("[BWD] no cached forward for session=%s", session_id)
            return
        tape, z_cut, z_mid = self._sessions.pop(session_id)
        dL_dz_mid = tf.convert_to_tensor(dL_dz_mid_np, dtype=tf.float32)
        targets = self.M2.trainable_variables + [z_cut]
        grads_all = tape.gradient(z_mid, targets, output_gradients=dL_dz_mid)
        grads_M2 = grads_all[:-1]
        dL_dz_cut = grads_all[-1]
        self.opt_M2.apply_gradients(zip(grads_M2, self.M2.trainable_variables))
        dL_dz_cut_np = dL_dz_cut.numpy()

        resp = self.encode_fn("BWD_RES", session_id, self.name, dL_dz_cut_np, reply_token=reply_token)
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
        self._infer_count += 1
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        z_mid = self.M2(z_cut, training=False)
        z_mid_np = z_mid.numpy()

        resp = self.encode_fn("INFER_RES", session_id, self.name, z_mid_np, reply_token=reply_token)
        self._sleep_send()
        if self.use_pir:
            self._pir_post(resp)
        else:
            self.board.update_bucket(bucket_id, resp)
        self.bytes_sent += len(resp)


class BaseBucketM1M3Peer(BasePeer):
    """Shared training/eval loop for M1+M3 peers using bucket Board API."""

    def __init__(
        self,
        name: str,
        run_dir: str,
        board_client: BucketBoardClient,
        M1,
        M3,
        opt_M1: optimizers.Optimizer,
        opt_M3: optimizers.Optimizer,
        loss_fn,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        target_m2: str,
        encode_fn: EncodeFn,
        decode_fn: DecodeWithHeaderFn,
        batch_size: int,
        epochs: int,
        lr: float,
        logger: logging.Logger,
        max_steps_per_epoch: int = 0,
        verbose: bool = False,
        log_every: int = 50,
        replay_protection: bool = True,
        replay_cache_size: int = 10000,
        replay_window_ms: int = 300000,
        replay_future_ms: int = 60000,
        send_jitter_ms: int = 0,
        poll_jitter_ms: int = 0,
        dummy_rate: float = 0.0,
        bucket_namespace: str = "",
        pseudonym_scope: str = "per_run",
        poll_shuffle: bool = False,
        use_pir: bool = False,
        pir_client: BasePIRBoardClient | None = None,
        pir_key: str = "",
        pir_chunk_size: int = 32,
        pir_clue_limit: int = 0,
        pir_rotation_seconds: int = 0,
        pir_rotation_grace: int = 1,
    ):
        super().__init__(
            name=name,
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
            use_pir=use_pir,
            pir_client=pir_client,
            pir_key=pir_key,
            pir_chunk_size=pir_chunk_size,
            pir_clue_limit=pir_clue_limit,
            pir_rotation_seconds=pir_rotation_seconds,
            pir_rotation_grace=pir_rotation_grace,
        )
        self.run_dir = run_dir
        self.board = board_client
        self.M1 = M1
        self.M3 = M3
        self.opt_M1 = opt_M1
        self.opt_M3 = opt_M3
        self.loss_fn = loss_fn
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.target_m2 = target_m2
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.bucket_namespace = bucket_namespace or ""
        self.pseudonym_scope = pseudonym_scope
        self.poll_shuffle = bool(poll_shuffle)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_steps_per_epoch = max(0, int(max_steps_per_epoch))
        self._seen_buckets: set[str] = set()
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0
        self._session_pseudos: dict[str, str] = {}
        self._session_tokens: dict[str, str] = {}
        self._pending: dict[tuple[str, str], np.ndarray] = {}

    def _wait_for(
        self,
        bucket_id: str,
        session_id: str,
        expect_op: str,
        reply_token: str | None = None,
    ) -> np.ndarray:
        if self.use_pir:
            token = reply_token or session_id
            cached = self._pending.pop((token, expect_op), None)
            if cached is not None:
                return cached
            while True:
                item = self._pir_next_payload()
                if not item:
                    self._sleep_poll()
                    continue
                idx, payload = item
                self.bytes_received += len(payload)
                try:
                    op, sess, sender, tensor, header = self.decode_fn(payload)
                except Exception:
                    self._pir_retry_index(idx)
                    continue
                if self._is_replay(header):
                    self._pir_mark_retrieved(idx)
                    continue
                if op == "DUMMY":
                    self._pir_mark_retrieved(idx)
                    continue
                msg_token = header.get("reply_token") or sess
                if op == expect_op and msg_token == token:
                    self._pir_mark_retrieved(idx)
                    return tensor
                self._pending[(msg_token, op)] = tensor
                self._pir_mark_retrieved(idx)
            return None
        while True:
            buckets = self.board.poll_buckets(namespace=self.bucket_namespace)
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
                    op, sess, sender, tensor, header = self.decode_fn(b["payload"])
                except Exception:
                    continue
                if self._is_replay(header):
                    continue
                if op == expect_op and sess == session_id:
                    if reply_token and header.get("reply_token") != reply_token:
                        continue
                    return tensor
            self._sleep_poll()

    def _sender_for_session(self, session_id: str) -> str:
        if self.pseudonym_scope != "per_session":
            return ""
        pseudo = self._session_pseudos.get(session_id)
        if not pseudo:
            pseudo = f"cli_{uuid.uuid4().hex[:8]}"
            self._session_pseudos[session_id] = pseudo
        return pseudo

    def _reply_token_for_session(self, session_id: str) -> str:
        token = self._session_tokens.get(session_id)
        if not token:
            token = uuid.uuid4().hex
            self._session_tokens[session_id] = token
        return token

    def _maybe_send_dummy(self):
        if self.dummy_rate <= 0.0:
            return
        if random.random() >= self.dummy_rate:
            return
        try:
            dummy_session = f"{self.name}-dummy-{uuid.uuid4().hex}"
            dummy_tensor = np.zeros((0,), dtype=np.float32)
            sender_pseudo = self._sender_for_session(dummy_session)
            payload = self.encode_fn("DUMMY", dummy_session, sender_pseudo, dummy_tensor)
            if self.use_pir:
                self._pir_post(payload)
            else:
                self.board.create_bucket(payload, namespace=self.bucket_namespace)
            self.bytes_sent += len(payload)
        except Exception:
            pass

    def train_batches(self, write_metrics: Callable[[int, int, float, float], None]):
        n = len(self.x_train)
        steps_per_epoch = n // self.batch_size
        if self.max_steps_per_epoch and self.max_steps_per_epoch < steps_per_epoch:
            steps_per_epoch = self.max_steps_per_epoch
        self.logger.info("Starting training on %d samples, %d steps/epoch (bucket).", n, steps_per_epoch)

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

                with tf.GradientTape(persistent=True) as tape_M1:
                    z_cut = self.M1(xb, training=True)
                z_cut_np = z_cut.numpy()
                session_id = f"{self.name}-train-{epoch}-{step}-{uuid.uuid4().hex}"

                self._fwd_count += 1
                self._sleep_send()
                sender_pseudo = self._sender_for_session(session_id)
                reply_token = self._reply_token_for_session(session_id)
                payload = self.encode_fn("FWD_REQ", session_id, sender_pseudo, z_cut_np, reply_token=reply_token)
                if self.use_pir:
                    bucket_id = ""
                    self._pir_post(payload)
                else:
                    bucket_id = self.board.create_bucket(payload, namespace=self.bucket_namespace)
                self.bytes_sent += len(payload)
                self._maybe_send_dummy()

                z_mid_np = self._wait_for(bucket_id, session_id, "FWD_RES", reply_token=reply_token)
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
                self._sleep_send()
                sender_pseudo = self._sender_for_session(session_id)
                payload = self.encode_fn("BWD_REQ", session_id, sender_pseudo, dL_dz_mid_np, reply_token=reply_token)
                if self.use_pir:
                    self._pir_post(payload)
                else:
                    self.board.update_bucket(bucket_id, payload)
                self.bytes_sent += len(payload)
                self._maybe_send_dummy()

                dL_dz_cut_np = self._wait_for(bucket_id, session_id, "BWD_RES", reply_token=reply_token)
                if not self.use_pir:
                    self.board.ack_bucket(bucket_id)

                dL_dz_cut = tf.convert_to_tensor(dL_dz_cut_np, dtype=tf.float32)
                grads_M1 = tape_M1.gradient(z_cut, self.M1.trainable_variables, output_gradients=dL_dz_cut)
                self.opt_M1.apply_gradients(zip(grads_M1, self.M1.trainable_variables))
                del tape_M1

                acc_batch = float(np.mean(np.argmax(logits.numpy(), axis=1) == yb.numpy()))
                loss_val = float(loss_value.numpy())
                epoch_losses.append(loss_val)
                epoch_accs.append(acc_batch)
                write_metrics(epoch, step, loss_val, acc_batch)

                if step % 100 == 0:
                    self.logger.info(
                        "Epoch %d Step %d/%d Loss=%.4f Acc=%.4f",
                        epoch,
                        step,
                        steps_per_epoch,
                        loss_val,
                        acc_batch,
                    )

            mean_loss = float(np.mean(epoch_losses))
            mean_acc = float(np.mean(epoch_accs))
            elapsed = time.time() - start
            self.logger.info(
                "Epoch %d done in %.1fs -> Loss=%.4f Acc=%.4f", epoch, elapsed, mean_loss, mean_acc
            )
            self.logger.info(
                "[bytes-epoch] epoch=%d sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
                epoch,
                self.bytes_sent - sent_start,
                self.bytes_received - recv_start,
                self._fwd_count - fwd_start,
                self._bwd_count - bwd_start,
                self._infer_count - infer_start,
            )

        self.logger.info(
            "[bytes-summary] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
            self.bytes_sent,
            self.bytes_received,
            self._fwd_count,
            self._bwd_count,
            self._infer_count,
        )

    def evaluate_batches(self) -> float:
        batch_size = self.batch_size
        n = len(self.x_test)
        all_logits = []

        for i in range(0, n, batch_size):
            xb = tf.convert_to_tensor(self.x_test[i:i + batch_size], dtype=tf.float32)
            z_cut = self.M1(xb, training=False)
            z_cut_np = z_cut.numpy()
            session_id = f"{self.name}-eval-{i}-{uuid.uuid4().hex}"

            self._infer_count += 1
            self._sleep_send()
            sender_pseudo = self._sender_for_session(session_id)
            reply_token = self._reply_token_for_session(session_id)
            payload = self.encode_fn("INFER_REQ", session_id, sender_pseudo, z_cut_np, reply_token=reply_token)
            if self.use_pir:
                bucket_id = ""
                self._pir_post(payload)
            else:
                bucket_id = self.board.create_bucket(payload, namespace=self.bucket_namespace)
            self.bytes_sent += len(payload)
            self._maybe_send_dummy()

            z_mid_np = self._wait_for(bucket_id, session_id, "INFER_RES", reply_token=reply_token)
            if not self.use_pir:
                self.board.ack_bucket(bucket_id)

            z_mid = tf.convert_to_tensor(z_mid_np, dtype=tf.float32)
            logits = self.M3(z_mid, training=False)
            all_logits.append(logits.numpy())

        logits_full = np.concatenate(all_logits, axis=0)[:n]
        acc = float(np.mean(np.argmax(logits_full, axis=1) == self.y_test))
        self.logger.info(
            "[bytes-summary] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
            self.bytes_sent,
            self.bytes_received,
            self._fwd_count,
            self._bwd_count,
            self._infer_count,
        )
        return acc
