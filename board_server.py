#!/usr/bin/env python3
"""
gRPC Board server for OMRsplit.

This exposes the same routing semantics as the in-process Ray Board actor:
  - PostMessage(sender, receiver, payload) -> msg_id
  - PollMessage(receiver) -> earliest queued message or empty

Messages are opaque byte blobs; payload structure and encryption are owned by peers.
"""

from concurrent import futures
import argparse
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque

import grpc

import board_pb2
import board_pb2_grpc
from pir_utils import (
    bytes_to_chunks,
    enc_number_from_bytes,
    enc_number_to_bytes,
    public_key_from_bytes,
)


class BoardStore:
    """
    Thread-safe in-memory message queues.

    - Legacy mode: per-receiver queue (board-blind).
    - Pool mode: named pools (single-blind) keyed by 'audience' strings.
    """

    def __init__(self, logger: logging.Logger, verbose: bool = False, log_every: int = 100):
        self.logger = logger
        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self._post_count = 0
        self._poll_count = 0
        self._lock = threading.Lock()
        self._queues: dict[str, deque] = defaultdict(deque)
        self._pools: dict[str, deque] = defaultdict(deque)
        self._pool_index: dict[str, str] = {}  # msg_id -> audience
        self._bytes_posted = 0
        self._bytes_polled = 0
        self._sender_counts: dict[str, int] = defaultdict(int)
        self._sender_bytes: dict[str, int] = defaultdict(int)
        self._receiver_counts: dict[str, int] = defaultdict(int)
        self._receiver_bytes: dict[str, int] = defaultdict(int)
        self._audience_counts: dict[str, int] = defaultdict(int)
        self._audience_bytes: dict[str, int] = defaultdict(int)

    def post_message(self, sender: str, receiver: str, payload: bytes, audience: str | None = None) -> str:
        with self._lock:
            self._post_count += 1
            msg_id = uuid.uuid4().hex
            message = {
                "msg_id": msg_id,
                "sender": sender,
                "receiver": receiver,
                "payload": payload,
                "timestamp_ms": int(time.time() * 1000),
            }
            self._bytes_posted += len(payload)
            self._sender_counts[sender] += 1
            self._sender_bytes[sender] += len(payload)
            if audience:
                self._pools[audience].append(message)
                self._pool_index[msg_id] = audience
                self._audience_counts[audience] += 1
                self._audience_bytes[audience] += len(payload)
            else:
                self._queues[receiver].append(message)
                self._receiver_counts[receiver] += 1
                self._receiver_bytes[receiver] += len(payload)

        if self.verbose and (self._post_count % self.log_every == 0):
            self.logger.info(
                "POST #%s sender=%s receiver=%s audience=%s size=%dB msg_id=%s total_bytes_posted=%d",
                self._post_count,
                sender,
                receiver,
                audience or "",
                len(payload),
                msg_id,
                self._bytes_posted,
            )
            self.logger.info(
                "POST-STATS senders=%s receivers=%s audiences=%s sender_bytes=%s receiver_bytes=%s audience_bytes=%s",
                dict(self._sender_counts),
                dict(self._receiver_counts),
                dict(self._audience_counts),
                dict(self._sender_bytes),
                dict(self._receiver_bytes),
                dict(self._audience_bytes),
            )
        return msg_id

    def poll_message(self, receiver: str):
        with self._lock:
            self._poll_count += 1
            queue = self._queues.get(receiver)
            if not queue:
                return None
            msg = queue.popleft()
            self._bytes_polled += len(msg["payload"])

        if self.verbose and (self._poll_count % self.log_every == 0):
            self.logger.info(
                "POLL #%s receiver=%s msg_id=%s size=%dB total_bytes_polled=%d",
                self._poll_count,
                receiver,
                msg["msg_id"],
                len(msg["payload"]),
                self._bytes_polled,
            )
        return msg

    def poll_pool(self, audience: str, limit_count: int | None = None):
        with self._lock:
            queue = self._pools.get(audience)
            if not queue:
                return []
            msgs = list(queue)
            self._bytes_polled += sum(len(m["payload"]) for m in msgs)
        if self.verbose and msgs and (len(msgs) % self.log_every == 0):
            self.logger.info(
                "POLL-POOL audience=%s count=%d total_bytes_polled=%d",
                audience,
                len(msgs),
                self._bytes_polled,
            )
        return msgs

    def ack_message(self, msg_id: str, audience: str | None = None) -> bool:
        with self._lock:
            aud = audience or self._pool_index.get(msg_id)
            if not aud:
                return False
            queue = self._pools.get(aud)
            if not queue:
                return False
            removed = False
            new_q = deque()
            while queue:
                m = queue.popleft()
                if m["msg_id"] == msg_id:
                    removed = True
                else:
                    new_q.append(m)
            self._pools[aud] = new_q
            if removed and msg_id in self._pool_index:
                del self._pool_index[msg_id]
            return removed

    def snapshot_metrics(self) -> dict:
        with self._lock:
            queue_count = sum(len(q) for q in self._queues.values())
            pool_count = sum(len(q) for q in self._pools.values())
            queue_bytes = sum(len(m["payload"]) for q in self._queues.values() for m in q)
            pool_bytes = sum(len(m["payload"]) for q in self._pools.values() for m in q)
        return {
            "queue_count": queue_count,
            "pool_count": pool_count,
            "queue_bytes": queue_bytes,
            "pool_bytes": pool_bytes,
        }

class BoardService(board_pb2_grpc.BoardServiceServicer):
    """gRPC servicer that forwards to BoardStore."""

    def __init__(self, store: BoardStore, logger: logging.Logger):
        self.store = store
        self.logger = logger

    def PostMessage(self, request: board_pb2.PostMessageRequest, context):
        audience = request.audience if request.audience else None
        msg_id = self.store.post_message(request.sender, request.receiver, request.payload, audience=audience)
        self.logger.info(
            "RPC PostMessage sender=%s receiver=%s audience=%s size=%dB msg_id=%s",
            request.sender,
            request.receiver,
            audience or "",
            len(request.payload),
            msg_id,
        )
        return board_pb2.PostMessageResponse(msg_id=msg_id)

    def PollMessage(self, request: board_pb2.PollMessageRequest, context):
        msg = self.store.poll_message(request.receiver)
        if not msg:
            self.logger.info("RPC PollMessage receiver=%s -> empty", request.receiver)
            return board_pb2.PollMessageResponse(has_message=False)

        proto_msg = board_pb2.Message(
            msg_id=msg["msg_id"],
            sender=msg["sender"],
            receiver=msg["receiver"],
            payload=msg["payload"],
            timestamp_ms=msg["timestamp_ms"],
        )
        self.logger.info(
            "RPC PollMessage receiver=%s -> msg_id=%s size=%dB from=%s",
            request.receiver,
            msg["msg_id"],
            len(msg["payload"]),
            msg["sender"],
        )
        return board_pb2.PollMessageResponse(has_message=True, message=proto_msg)

    def PollPool(self, request: board_pb2.PollPoolRequest, context):
        msgs = self.store.poll_pool(request.audience, limit_count=None)
        if not msgs:
            self.logger.info("RPC PollPool audience=%s -> empty", request.audience)
            return board_pb2.PollPoolResponse(messages=[])

        proto_msgs = [
            board_pb2.Message(
                msg_id=m["msg_id"],
                sender=m["sender"],
                receiver=m["receiver"],
                payload=m["payload"],
                timestamp_ms=m["timestamp_ms"],
            )
            for m in msgs
        ]
        self.logger.info(
            "RPC PollPool audience=%s -> count=%d", request.audience, len(proto_msgs)
        )
        return board_pb2.PollPoolResponse(messages=proto_msgs)

    def AckMessage(self, request: board_pb2.AckMessageRequest, context):
        removed = self.store.ack_message(request.msg_id, audience=request.audience or None)
        self.logger.info("RPC AckMessage audience=%s msg_id=%s removed=%s", request.audience, request.msg_id, removed)
        return board_pb2.AckMessageResponse(removed=removed)


# ============================================================
# Bucket mode
# ============================================================

class BucketBoardStore:
    def __init__(self, logger: logging.Logger, verbose: bool = False, log_every: int = 100):
        self.logger = logger
        self.verbose = verbose
        self.log_every = max(1, log_every)
        self._lock = threading.Lock()
        self._buckets: dict[str, dict[str, dict]] = defaultdict(dict)
        self._bucket_index: dict[str, str] = {}  # bucket_id -> namespace
        self._create_count = 0
        self._poll_count = 0
        self._update_count = 0
        self._bytes_created = 0
        self._bytes_polled = 0
        self._bytes_updated = 0

    def create_bucket(self, payload: bytes, namespace: str = "") -> str:
        with self._lock:
            self._create_count += 1
            bucket_id = uuid.uuid4().hex
            self._buckets[namespace][bucket_id] = {
                "bucket_id": bucket_id,
                "payload": payload,
                "timestamp_ms": int(time.time() * 1000),
                "namespace": namespace,
            }
            self._bucket_index[bucket_id] = namespace
            self._bytes_created += len(payload)
        if self.verbose and (self._create_count % self.log_every == 0):
            self.logger.info(
                "CREATE #%s bucket_id=%s namespace=%s size=%dB total_bytes_created=%d",
                self._create_count,
                bucket_id,
                namespace,
                len(payload),
                self._bytes_created,
            )
        return bucket_id

    def poll_buckets(self, namespace: str | None = None):
        with self._lock:
            self._poll_count += 1
            if namespace is None or namespace == "":
                buckets = [b for ns in self._buckets.values() for b in ns.values()]
            else:
                buckets = list(self._buckets.get(namespace, {}).values())
            self._bytes_polled += sum(len(b["payload"]) for b in buckets)
        if self.verbose and buckets and (self._poll_count % self.log_every == 0):
            self.logger.info(
                "POLL #%s namespace=%s count=%d total_bytes_polled=%d",
                self._poll_count,
                namespace or "",
                len(buckets),
                self._bytes_polled,
            )
        return buckets

    def update_bucket(self, bucket_id: str, payload: bytes) -> bool:
        with self._lock:
            self._update_count += 1
            namespace = self._bucket_index.get(bucket_id, "")
            bucket_map = self._buckets.get(namespace)
            if not bucket_map or bucket_id not in bucket_map:
                return False
            bucket_map[bucket_id] = {
                "bucket_id": bucket_id,
                "payload": payload,
                "timestamp_ms": int(time.time() * 1000),
                "namespace": namespace,
            }
            self._bytes_updated += len(payload)
        if self.verbose and (self._update_count % self.log_every == 0):
            self.logger.info(
                "UPDATE #%s bucket_id=%s namespace=%s size=%dB total_bytes_updated=%d",
                self._update_count,
                bucket_id,
                namespace,
                len(payload),
                self._bytes_updated,
            )
        return True

    def ack_bucket(self, bucket_id: str) -> bool:
        with self._lock:
            namespace = self._bucket_index.get(bucket_id, "")
            bucket_map = self._buckets.get(namespace)
            if not bucket_map:
                return False
            removed = bucket_id in bucket_map
            bucket_map.pop(bucket_id, None)
            self._bucket_index.pop(bucket_id, None)
            return removed

    def snapshot_metrics(self) -> dict:
        with self._lock:
            buckets = [b for ns in self._buckets.values() for b in ns.values()]
            bucket_count = len(buckets)
            bucket_bytes = sum(len(b["payload"]) for b in buckets)
        return {
            "bucket_count": bucket_count,
            "bucket_bytes": bucket_bytes,
        }


class BucketBoardService(board_pb2_grpc.BucketBoardServicer):
    def __init__(self, store: BucketBoardStore, logger: logging.Logger):
        self.store = store
        self.logger = logger

    def CreateBucket(self, request, context):
        namespace = request.namespace if hasattr(request, "namespace") else ""
        bucket_id = self.store.create_bucket(request.payload, namespace=namespace)
        self.logger.info(
            "RPC CreateBucket bucket_id=%s namespace=%s size=%dB",
            bucket_id,
            namespace,
            len(request.payload),
        )
        return board_pb2.CreateBucketResponse(bucket_id=bucket_id)

    def PollBuckets(self, request, context):
        namespace = request.namespace if hasattr(request, "namespace") else ""
        buckets = self.store.poll_buckets(namespace=namespace)
        resp_buckets = [
            board_pb2.Bucket(
                bucket_id=b["bucket_id"],
                payload=b["payload"],
                timestamp_ms=b["timestamp_ms"],
                namespace=b.get("namespace", ""),
            )
            for b in buckets
        ]
        self.logger.info(
            "RPC PollBuckets namespace=%s -> count=%d",
            namespace,
            len(resp_buckets),
        )
        return board_pb2.PollBucketsResponse(buckets=resp_buckets)

    def UpdateBucket(self, request, context):
        updated = self.store.update_bucket(request.bucket_id, request.payload)
        self.logger.info(
            "RPC UpdateBucket bucket_id=%s updated=%s size=%dB", request.bucket_id, updated, len(request.payload)
        )
        return board_pb2.UpdateBucketResponse(updated=updated)

    def AckBucket(self, request, context):
        removed = self.store.ack_bucket(request.bucket_id)
        self.logger.info("RPC AckBucket bucket_id=%s removed=%s", request.bucket_id, removed)
        return board_pb2.AckBucketResponse(removed=removed)


class PIRStore:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._lock = threading.Lock()
        self._payloads: list[bytes] = []
        self._clues: list[bytes] = []
        self._lengths: list[int] = []

    def post(self, payload: bytes, clue: bytes) -> tuple[int, int]:
        with self._lock:
            index = len(self._payloads)
            self._payloads.append(payload)
            self._clues.append(clue)
            self._lengths.append(len(payload))
            total = len(self._payloads)
        return index, total

    def get_clues(self, start_index: int = 0, limit: int = 0):
        with self._lock:
            total = len(self._payloads)
            start = max(0, int(start_index))
            end = total if limit <= 0 else min(total, start + int(limit))
            clues = [
                {
                    "index": i,
                    "clue": self._clues[i],
                    "payload_len": self._lengths[i],
                }
                for i in range(start, end)
            ]
        latest = total - 1 if total > 0 else 0
        return clues, latest, total

    def snapshot_payloads(self):
        with self._lock:
            payloads = list(self._payloads)
            lengths = list(self._lengths)
        return payloads, lengths

    def snapshot_metrics(self) -> dict:
        with self._lock:
            payload_bytes = sum(len(p) for p in self._payloads)
            clue_bytes = sum(len(c) for c in self._clues)
            count = len(self._payloads)
        return {
            "pir_count": count,
            "pir_payload_bytes": payload_bytes,
            "pir_clue_bytes": clue_bytes,
        }


class PIRService(board_pb2_grpc.PIRServiceServicer):
    def __init__(self, store: PIRStore, logger: logging.Logger):
        self.store = store
        self.logger = logger

    def PostPIRMessage(self, request: board_pb2.PIRPostRequest, context):
        index, total = self.store.post(request.payload, request.clue)
        self.logger.info("RPC PostPIRMessage index=%d size=%dB total=%d", index, len(request.payload), total)
        return board_pb2.PIRPostResponse(index=index, total=total)

    def GetClues(self, request: board_pb2.PIRCluesRequest, context):
        clues, latest, total = self.store.get_clues(request.start_index, request.limit)
        resp_clues = [
            board_pb2.PIRClue(index=c["index"], clue=c["clue"], payload_len=c["payload_len"])
            for c in clues
        ]
        self.logger.info(
            "RPC GetClues start=%d limit=%d -> count=%d total=%d",
            request.start_index,
            request.limit,
            len(resp_clues),
            total,
        )
        return board_pb2.PIRCluesResponse(clues=resp_clues, latest_index=latest, total=total)

    def PIRQuery(self, request: board_pb2.PIRQueryRequest, context):
        try:
            public_key = public_key_from_bytes(request.public_key)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid public key: {exc}")
            return board_pb2.PIRQueryResponse()

        selector = [enc_number_from_bytes(public_key, b) for b in request.enc_selector]
        payloads, lengths = self.store.snapshot_payloads()
        if len(selector) > len(payloads):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Selector length is larger than database size.")
            return board_pb2.PIRQueryResponse()

        if not payloads:
            return board_pb2.PIRQueryResponse(enc_chunks=[], record_len=0, total=0)

        if len(selector) < len(payloads):
            zero_ct = selector[0] * 0
            selector = selector + [zero_ct for _ in range(len(payloads) - len(selector))]

        chunk_size = max(1, int(request.chunk_size))
        record_len = max(lengths) if lengths else 0
        num_chunks = (record_len + chunk_size - 1) // chunk_size

        chunked = [
            bytes_to_chunks(payloads[i], chunk_size, total_len=record_len)
            for i in range(len(payloads))
        ]

        enc_chunks = []
        for j in range(num_chunks):
            enc_sum = None
            for i, enc_sel in enumerate(selector):
                val = chunked[i][j]
                if val == 0:
                    continue
                term = enc_sel * val
                enc_sum = term if enc_sum is None else enc_sum + term
            if enc_sum is None:
                enc_sum = selector[0] * 0
            enc_chunks.append(enc_number_to_bytes(enc_sum))

        self.logger.info(
            "RPC PIRQuery size=%d record_len=%d chunk_size=%d chunks=%d",
            len(payloads),
            record_len,
            chunk_size,
            len(enc_chunks),
        )
        return board_pb2.PIRQueryResponse(enc_chunks=enc_chunks, record_len=record_len, total=len(payloads))


class BoardMetricsLogger:
    def __init__(
        self,
        path: str,
        interval: float,
        board_store: BoardStore,
        bucket_store: BucketBoardStore,
        pir_store: PIRStore,
        logger: logging.Logger,
    ):
        self.path = path
        self.interval = max(0.1, float(interval))
        self.board_store = board_store
        self.bucket_store = bucket_store
        self.pir_store = pir_store
        self.logger = logger
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(
                "ts_ms,elapsed_s,queue_bytes,pool_bytes,bucket_bytes,pir_payload_bytes,pir_clue_bytes,total_bytes\n"
            )
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        start = time.time()
        while not self._stop.is_set():
            now = time.time()
            board_metrics = self.board_store.snapshot_metrics()
            bucket_metrics = self.bucket_store.snapshot_metrics()
            pir_metrics = self.pir_store.snapshot_metrics()
            queue_bytes = int(board_metrics.get("queue_bytes", 0))
            pool_bytes = int(board_metrics.get("pool_bytes", 0))
            bucket_bytes = int(bucket_metrics.get("bucket_bytes", 0))
            pir_payload_bytes = int(pir_metrics.get("pir_payload_bytes", 0))
            pir_clue_bytes = int(pir_metrics.get("pir_clue_bytes", 0))
            total_bytes = queue_bytes + pool_bytes + bucket_bytes + pir_payload_bytes + pir_clue_bytes
            line = (
                f"{int(now * 1000)},{now - start:.3f},{queue_bytes},{pool_bytes},{bucket_bytes},"
                f"{pir_payload_bytes},{pir_clue_bytes},{total_bytes}\n"
            )
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as exc:
                self.logger.warning("Failed to write board metrics: %s", exc)
            self._stop.wait(self.interval)


def serve(
    host: str,
    port: int,
    log_level: str = "INFO",
    verbose: bool = False,
    log_every: int = 100,
    metrics_path: str | None = None,
    metrics_interval: float = 1.0,
):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("board_server")
    store = BoardStore(logger=logger, verbose=verbose, log_every=log_every)
    bucket_store = BucketBoardStore(logger=logger, verbose=verbose, log_every=log_every)
    pir_store = PIRStore(logger=logger)

    grpc_opts = [
        ("grpc.max_send_message_length", 128 * 1024 * 1024),
        ("grpc.max_receive_message_length", 128 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32), options=grpc_opts)
    board_pb2_grpc.add_BoardServiceServicer_to_server(BoardService(store, logger), server)
    board_pb2_grpc.add_BucketBoardServicer_to_server(BucketBoardService(bucket_store, logger), server)
    board_pb2_grpc.add_PIRServiceServicer_to_server(PIRService(pir_store, logger), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info("Board gRPC server listening on %s:%d (verbose=%s, log_every=%d)", host, port, verbose, log_every)
    metrics_logger = None
    if metrics_path:
        metrics_logger = BoardMetricsLogger(
            path=metrics_path,
            interval=metrics_interval,
            board_store=store,
            bucket_store=bucket_store,
            pir_store=pir_store,
            logger=logger,
        )
        metrics_logger.start()
    try:
        server.wait_for_termination()
    finally:
        if metrics_logger:
            metrics_logger.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the OMRsplit Board gRPC server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host/interface to bind.")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, etc.).")
    parser.add_argument("--verbose", action="store_true", help="Enable periodic POST/POLL logs.")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N POST/POLL calls when verbose.")
    parser.add_argument("--metrics-path", default="", help="Optional CSV path for board size metrics.")
    parser.add_argument("--metrics-interval", type=float, default=1.0, help="Metrics sampling interval in seconds.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    serve(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        verbose=args.verbose,
        log_every=args.log_every,
        metrics_path=args.metrics_path or None,
        metrics_interval=args.metrics_interval,
    )
