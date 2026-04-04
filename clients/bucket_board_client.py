import grpc

import board_pb2
import board_pb2_grpc


class BucketBoardClient:
    """Helper client for the bucket-based board API in board.proto."""

    def __init__(self, host: str, port: int):
        opts = [
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(f"{host}:{port}", options=opts)
        self.stub = board_pb2_grpc.BucketBoardStub(self.channel)

    def create_bucket(self, payload: bytes, namespace: str = "") -> str:
        resp = self.stub.CreateBucket(board_pb2.CreateBucketRequest(payload=payload, namespace=namespace))
        return resp.bucket_id

    def poll_buckets(self, namespace: str = ""):
        resp = self.stub.PollBuckets(board_pb2.PollBucketsRequest(namespace=namespace))
        return [
            {
                "bucket_id": b.bucket_id,
                "payload": bytes(b.payload),
                "timestamp_ms": b.timestamp_ms,
                "namespace": b.namespace if hasattr(b, "namespace") else "",
            }
            for b in resp.buckets
        ]

    def update_bucket(self, bucket_id: str, payload: bytes) -> bool:
        resp = self.stub.UpdateBucket(board_pb2.UpdateBucketRequest(bucket_id=bucket_id, payload=payload))
        return resp.updated

    def ack_bucket(self, bucket_id: str) -> bool:
        resp = self.stub.AckBucket(board_pb2.AckBucketRequest(bucket_id=bucket_id))
        return resp.removed
