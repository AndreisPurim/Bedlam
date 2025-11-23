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
    """Thread-safe in-memory message queues keyed by receiver."""

    def __init__(self, logger: logging.Logger, verbose: bool = False, log_every: int = 100):
        self.logger = logger
        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self._post_count = 0
        self._poll_count = 0
        self._lock = threading.Lock()
        self._queues: dict[str, deque] = defaultdict(deque)

    def post_message(self, sender: str, receiver: str, payload: bytes) -> str:
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
            self._queues[receiver].append(message)

        if self.verbose and (self._post_count % self.log_every == 0):
            self.logger.info(
                "POST #%s sender=%s receiver=%s size=%dB msg_id=%s",
                self._post_count,
                sender,
                receiver,
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


class BoardService(board_pb2_grpc.BoardServiceServicer):
    """gRPC servicer that forwards to BoardStore."""

    def __init__(self, store: BoardStore, logger: logging.Logger):
        self.store = store
        self.logger = logger

    def PostMessage(self, request: board_pb2.PostMessageRequest, context):
        msg_id = self.store.post_message(request.sender, request.receiver, request.payload)
        self.logger.info(
            "RPC PostMessage sender=%s receiver=%s size=%dB msg_id=%s",
            request.sender,
            request.receiver,
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


def serve(host: str, port: int, log_level: str = "INFO", verbose: bool = False, log_every: int = 100):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("board_server")
    store = BoardStore(logger=logger, verbose=verbose, log_every=log_every)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    board_pb2_grpc.add_BoardServiceServicer_to_server(BoardService(store, logger), server)
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
