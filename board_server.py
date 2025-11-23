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
import threading
import time
import uuid
from collections import defaultdict, deque

import grpc

import board_pb2
import board_pb2_grpc


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
            if audience:
                self._pools[audience].append(message)
                self._pool_index[msg_id] = audience
            else:
                self._queues[receiver].append(message)

        if self.verbose and (self._post_count % self.log_every == 0):
            self.logger.info(
                "POST #%s sender=%s receiver=%s audience=%s size=%dB msg_id=%s",
                self._post_count,
                sender,
                receiver,
                audience or "",
                len(payload),
                msg_id,
            )
        return msg_id

    def poll_message(self, receiver: str):
        with self._lock:
            self._poll_count += 1
            queue = self._queues.get(receiver)
            if not queue:
                return None
            msg = queue.popleft()

        if self.verbose and (self._poll_count % self.log_every == 0):
            self.logger.info(
                "POLL #%s receiver=%s msg_id=%s size=%dB",
                self._poll_count,
                receiver,
                msg["msg_id"],
                len(msg["payload"]),
            )
        return msg

    def poll_pool(self, audience: str, limit_count: int | None = None):
        with self._lock:
            queue = self._pools.get(audience)
            if not queue:
                return []
            msgs = list(queue)
        if self.verbose and msgs and (len(msgs) % self.log_every == 0):
            self.logger.info("POLL-POOL audience=%s count=%d", audience, len(msgs))
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
        self._buckets: dict[str, dict] = {}
        self._create_count = 0
        self._poll_count = 0
        self._update_count = 0

    def create_bucket(self, payload: bytes) -> str:
        self._create_count += 1
        bucket_id = uuid.uuid4().hex
        self._buckets[bucket_id] = {
            "bucket_id": bucket_id,
            "payload": payload,
            "timestamp_ms": int(time.time() * 1000),
        }
        if self.verbose and (self._create_count % self.log_every == 0):
            self.logger.info("CREATE #%s bucket_id=%s size=%dB", self._create_count, bucket_id, len(payload))
        return bucket_id

    def poll_buckets(self):
        self._poll_count += 1
        buckets = list(self._buckets.values())
        if self.verbose and buckets and (self._poll_count % self.log_every == 0):
            self.logger.info("POLL #%s count=%d", self._poll_count, len(buckets))
        return buckets

    def update_bucket(self, bucket_id: str, payload: bytes) -> bool:
        self._update_count += 1
        if bucket_id not in self._buckets:
            return False
        self._buckets[bucket_id] = {
            "bucket_id": bucket_id,
            "payload": payload,
            "timestamp_ms": int(time.time() * 1000),
        }
        if self.verbose and (self._update_count % self.log_every == 0):
            self.logger.info("UPDATE #%s bucket_id=%s size=%dB", self._update_count, bucket_id, len(payload))
        return True

    def ack_bucket(self, bucket_id: str) -> bool:
        removed = bucket_id in self._buckets
        self._buckets.pop(bucket_id, None)
        return removed


class BucketBoardService(board_pb2_grpc.BucketBoardServicer):
    def __init__(self, store: BucketBoardStore, logger: logging.Logger):
        self.store = store
        self.logger = logger

    def CreateBucket(self, request, context):
        bucket_id = self.store.create_bucket(request.payload)
        self.logger.info("RPC CreateBucket bucket_id=%s size=%dB", bucket_id, len(request.payload))
        return board_pb2.CreateBucketResponse(bucket_id=bucket_id)

    def PollBuckets(self, request, context):
        buckets = self.store.poll_buckets()
        resp_buckets = [
            board_pb2.Bucket(
                bucket_id=b["bucket_id"],
                payload=b["payload"],
                timestamp_ms=b["timestamp_ms"],
            )
            for b in buckets
        ]
        self.logger.info("RPC PollBuckets -> count=%d", len(resp_buckets))
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


def serve(host: str, port: int, log_level: str = "INFO", verbose: bool = False, log_every: int = 100):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("board_server")
    store = BoardStore(logger=logger, verbose=verbose, log_every=log_every)
    bucket_store = BucketBoardStore(logger=logger, verbose=verbose, log_every=log_every)

    grpc_opts = [
        ("grpc.max_send_message_length", 128 * 1024 * 1024),
        ("grpc.max_receive_message_length", 128 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32), options=grpc_opts)
    board_pb2_grpc.add_BoardServiceServicer_to_server(BoardService(store, logger), server)
    board_pb2_grpc.add_BucketBoardServicer_to_server(BucketBoardService(bucket_store, logger), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info("Board gRPC server listening on %s:%d (verbose=%s, log_every=%d)", host, port, verbose, log_every)
    server.wait_for_termination()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the OMRsplit Board gRPC server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host/interface to bind.")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, etc.).")
    parser.add_argument("--verbose", action="store_true", help="Enable periodic POST/POLL logs.")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N POST/POLL calls when verbose.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    serve(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        verbose=args.verbose,
        log_every=args.log_every,
    )
