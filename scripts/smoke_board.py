import os
import signal
import socket
import subprocess
import sys
import time

import grpc

import board_pb2
import board_pb2_grpc


HOST = "127.0.0.1"
PORT = int(os.environ.get("BOARD_SMOKE_PORT", "50055"))


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def main():
    proc = subprocess.Popen(
        [sys.executable, "board_server.py", "--host", HOST, "--port", str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        if not wait_for_port(HOST, PORT, timeout=10):
            raise RuntimeError("board_server did not start")

        channel = grpc.insecure_channel(f"{HOST}:{PORT}")
        board = board_pb2_grpc.BoardServiceStub(channel)
        bucket = board_pb2_grpc.BucketBoardStub(channel)

        # Pool test
        payload = b"hello"
        post = board.PostMessage(board_pb2.PostMessageRequest(
            sender="cli_x",
            receiver="",
            payload=payload,
            audience="to_m2",
        ))
        msgs = board.PollPool(board_pb2.PollPoolRequest(audience="to_m2")).messages
        assert any(m.msg_id == post.msg_id for m in msgs), "pool message missing"
        board.AckMessage(board_pb2.AckMessageRequest(msg_id=post.msg_id, audience="to_m2"))
        msgs2 = board.PollPool(board_pb2.PollPoolRequest(audience="to_m2")).messages
        assert all(m.msg_id != post.msg_id for m in msgs2), "pool ack failed"

        # Bucket namespace test
        ns = "ns1"
        ns2 = "ns2"
        bresp = bucket.CreateBucket(board_pb2.CreateBucketRequest(payload=b"data", namespace=ns))
        buckets_ns = bucket.PollBuckets(board_pb2.PollBucketsRequest(namespace=ns)).buckets
        assert any(b.bucket_id == bresp.bucket_id for b in buckets_ns), "namespace bucket missing"
        buckets_ns2 = bucket.PollBuckets(board_pb2.PollBucketsRequest(namespace=ns2)).buckets
        assert all(b.bucket_id != bresp.bucket_id for b in buckets_ns2), "namespace isolation failed"
        bucket.AckBucket(board_pb2.AckBucketRequest(bucket_id=bresp.bucket_id))

        print("smoke ok")
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()