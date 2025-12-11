from __future__ import annotations

import logging
import time
from typing import Callable, Tuple

import grpc
import numpy as np
import tensorflow as tf
from keras import optimizers

import board_pb2
import board_pb2_grpc
from clients.bucket_board_client import BucketBoardClient


EncodeFn = Callable[[str, str, str, np.ndarray], bytes]
DecodeFn = Callable[[bytes], Tuple[str, str, str, np.ndarray]]
DecodeWithHeaderFn = Callable[[bytes], Tuple[str, str, str, np.ndarray, dict]]


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


class BaseBucketBoardClient(BucketBoardClient):
    """Alias for clarity when injecting into bucket-based peers."""
    pass


class BasePeer:
    def __init__(self, name: str, logger: logging.Logger, verbose: bool = False, log_every: int = 50):
        self.name = name
        self.logger = logger
        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self.bytes_sent = 0
        self.bytes_received = 0


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
    ):
        super().__init__(name=name, logger=logger, verbose=verbose, log_every=log_every)
        self.board_client = board_client
        self.M2 = model
        self.opt_M2 = optimizer
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self._sessions: dict[str, tuple] = {}
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0

    def run(self):
        """Process messages addressed to this peer via the Board."""
        self.logger.info("M2 peer '%s' run() loop started.", self.name)
        while True:
            msg = self.board_client.poll_message(receiver=self.name)
            if not msg:
                time.sleep(0.01)
                continue
            self.bytes_received += len(msg["payload"])
            try:
                op, session, sender, tensor = self.decode_fn(msg["payload"])
            except Exception as exc:  # best-effort robustness
                self.logger.error("Failed to decode message: %s", exc)
                continue

            if op == "FWD_REQ":
                self._handle_forward(session, sender, tensor)
            elif op == "BWD_REQ":
                self._handle_backward(session, sender, tensor)
            elif op == "INFER_REQ":
                self._handle_infer(session, sender, tensor)
            else:
                self.logger.warning("Unknown op '%s' in session=%s from=%s", op, session, sender)

            total = self._fwd_count + self._bwd_count + self._infer_count
            if total and (total % self.log_every == 0):
                self.logger.info(
                    "[bytes] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
                    self.bytes_sent,
                    self.bytes_received,
                    self._fwd_count,
                    self._bwd_count,
                    self._infer_count,
                )

    def _handle_forward(self, session_id: str, sender: str, z_cut_np: np.ndarray):
        self._fwd_count += 1
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z_cut)
            z_mid = self.M2(z_cut, training=True)
        self._sessions[session_id] = (tape, z_cut, z_mid)
        z_mid_np = z_mid.numpy()

        payload = self.encode_fn("FWD_RES", session_id, self.name, z_mid_np)
        self.board_client.post_message(sender=self.name, receiver=sender, payload=payload)
        self.bytes_sent += len(payload)

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

    def _handle_backward(self, session_id: str, sender: str, dL_dz_mid_np: np.ndarray):
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

        payload = self.encode_fn("BWD_RES", session_id, self.name, dL_dz_cut_np)
        self.board_client.post_message(sender=self.name, receiver=sender, payload=payload)
        self.bytes_sent += len(payload)

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

    def _handle_infer(self, session_id: str, sender: str, z_cut_np: np.ndarray):
        self._infer_count += 1
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        z_mid = self.M2(z_cut, training=False)
        z_mid_np = z_mid.numpy()

        payload = self.encode_fn("INFER_RES", session_id, self.name, z_mid_np)
        self.board_client.post_message(sender=self.name, receiver=sender, payload=payload)
        self.bytes_sent += len(payload)

        if self.verbose and (self._infer_count % self.log_every == 0):
            bs = z_cut_np.shape[0]
            self.logger.info(
                "[INFER #%d] session=%s from=%s batch=%d", self._infer_count, session_id, sender, bs
            )


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
        verbose: bool = False,
        log_every: int = 50,
    ):
        super().__init__(name=name, logger=logger, verbose=verbose, log_every=log_every)
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
        self._seen_msg_ids: set[str] = set()
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0

        self.metrics_path = None  # set by subclass when ready

    # The train/evaluate loops remain identical between vanilla and encrypted modes.
    def train_batches(self, write_metrics: Callable[[int, int, float, float], None]):
        n = len(self.x_train)
        steps_per_epoch = n // self.batch_size
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
                lo = step * self.batch_size
                hi = lo + self.batch_size
                xb = tf.convert_to_tensor(x_sh[lo:hi], dtype=tf.float32)
                yb = tf.convert_to_tensor(y_sh[lo:hi], dtype=tf.int32)

                with tf.GradientTape(persistent=True) as tape_M1:
                    z_cut = self.M1(xb, training=True)
                z_cut_np = z_cut.numpy()
                session_id = f"{self.name}-train-{epoch}-{step}"

                # Send forward request
                self._fwd_count += 1
                payload = self.encode_fn("FWD_REQ", session_id, self.sender_id, z_cut_np)
                self.board_client.post_message(sender=self.sender_id, receiver=self.target_m2, payload=payload)
                self.bytes_sent += len(payload)
                if self.verbose and (self._fwd_count % self.log_every == 0):
                    self.logger.info(
                        "[FWD_REQ #%d] session=%s to=%s batch=%d",
                        self._fwd_count,
                        session_id,
                        self.target_m2,
                        z_cut_np.shape[0],
                    )

                z_mid_np = self._wait_for(session_id, expect_op="FWD_RES")
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
                payload = self.encode_fn("BWD_REQ", session_id, self.sender_id, dL_dz_mid_np)
                self.board_client.post_message(sender=self.sender_id, receiver=self.target_m2, payload=payload)
                self.bytes_sent += len(payload)
                if self.verbose and (self._bwd_count % self.log_every == 0):
                    self.logger.info("[BWD_REQ #%d] session=%s to=%s", self._bwd_count, session_id, self.target_m2)

                dL_dz_cut_np = self._wait_for(session_id, expect_op="BWD_RES")
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
                "Epoch %d done in %.1fs → Loss=%.4f Acc=%.4f", epoch, elapsed, mean_loss, mean_acc
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
            session_id = f"{self.name}-eval-{i}"

            self._infer_count += 1
            payload = self.encode_fn("INFER_REQ", session_id, self.sender_id, z_cut_np)
            self.board_client.post_message(sender=self.sender_id, receiver=self.target_m2, payload=payload)
            self.bytes_sent += len(payload)
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

    def _wait_for(self, session_id: str, expect_op: str) -> np.ndarray:
        while True:
            msg = self.board_client.poll_message(receiver=self.sender_id)
            if not msg:
                time.sleep(0.01)
                continue
            msg_id = msg["msg_id"]
            if msg_id in self._seen_msg_ids:
                continue
            self._seen_msg_ids.add(msg_id)
            self.bytes_received += len(msg["payload"])
            try:
                op, sess, sender_name, tensor = self.decode_fn(msg["payload"])
            except Exception:
                continue
            if op == expect_op and sess == session_id:
                return tensor
            time.sleep(0.01)


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
    ):
        super().__init__(name=name, logger=logger, verbose=verbose, log_every=log_every)
        self.board = board_client
        self.M2 = model
        self.opt_M2 = optimizer
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.target_name = target_name
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

        if op == "FWD_REQ":
            self._handle_forward(bid, session, sender, tensor)
        elif op == "BWD_REQ":
            self._handle_backward(bid, session, sender, tensor)
        elif op == "INFER_REQ":
            self._handle_infer(bid, session, sender, tensor)

    def run(self):
        self.logger.info("Bucket M2 '%s' run() loop started.", self.name)
        while True:
            buckets = self.board.poll_buckets()
            if not buckets:
                time.sleep(0.01)
                continue
            for b in buckets:
                self.process_bucket(b)
            total = self._fwd_count + self._bwd_count + self._infer_count
            if total and (total % self.log_every == 0):
                self.logger.info(
                    "[bytes] sent=%dB recv=%dB fwd=%d bwd=%d infer=%d",
                    self.bytes_sent,
                    self.bytes_received,
                    self._fwd_count,
                    self._bwd_count,
                    self._infer_count,
                )

    # ----- handlers -----
    def _handle_forward(self, bucket_id: str, session_id: str, sender: str, z_cut_np: np.ndarray):
        self._fwd_count += 1
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z_cut)
            z_mid = self.M2(z_cut, training=True)
        self._sessions[session_id] = (tape, z_cut, z_mid)
        z_mid_np = z_mid.numpy()

        resp = self.encode_fn("FWD_RES", session_id, self.name, z_mid_np)
        self.board.update_bucket(bucket_id, resp)
        self.bytes_sent += len(resp)

    def _handle_backward(self, bucket_id: str, session_id: str, sender: str, dL_dz_mid_np: np.ndarray):
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

        resp = self.encode_fn("BWD_RES", session_id, self.name, dL_dz_cut_np)
        self.board.update_bucket(bucket_id, resp)
        self.bytes_sent += len(resp)

    def _handle_infer(self, bucket_id: str, session_id: str, sender: str, z_cut_np: np.ndarray):
        self._infer_count += 1
        z_cut = tf.convert_to_tensor(z_cut_np, dtype=tf.float32)
        z_mid = self.M2(z_cut, training=False)
        z_mid_np = z_mid.numpy()

        resp = self.encode_fn("INFER_RES", session_id, self.name, z_mid_np)
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
        verbose: bool = False,
        log_every: int = 50,
    ):
        super().__init__(name=name, logger=logger, verbose=verbose, log_every=log_every)
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
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self._seen_buckets: set[str] = set()
        self._fwd_count = 0
        self._bwd_count = 0
        self._infer_count = 0

    def _wait_for(self, bucket_id: str, session_id: str, expect_op: str) -> np.ndarray:
        while True:
            buckets = self.board.poll_buckets()
            if not buckets:
                time.sleep(0.01)
                continue
            for b in buckets:
                if b["bucket_id"] != bucket_id:
                    continue
                self.bytes_received += len(b["payload"])
                try:
                    op, sess, sender, tensor, header = self.decode_fn(b["payload"])
                except Exception:
                    continue
                if op == expect_op and sess == session_id:
                    return tensor
            time.sleep(0.01)

    def train_batches(self, write_metrics: Callable[[int, int, float, float], None]):
        n = len(self.x_train)
        steps_per_epoch = n // self.batch_size
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
                session_id = f"{self.name}-train-{epoch}-{step}"

                self._fwd_count += 1
                payload = self.encode_fn("FWD_REQ", session_id, "", z_cut_np)
                bucket_id = self.board.create_bucket(payload)
                self.bytes_sent += len(payload)

                z_mid_np = self._wait_for(bucket_id, session_id, "FWD_RES")
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
                payload = self.encode_fn("BWD_REQ", session_id, "", dL_dz_mid_np)
                self.board.update_bucket(bucket_id, payload)
                self.bytes_sent += len(payload)

                dL_dz_cut_np = self._wait_for(bucket_id, session_id, "BWD_RES")
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
                "Epoch %d done in %.1fs → Loss=%.4f Acc=%.4f", epoch, elapsed, mean_loss, mean_acc
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
            session_id = f"{self.name}-eval-{i}"

            self._infer_count += 1
            payload = self.encode_fn("INFER_REQ", session_id, "", z_cut_np)
            bucket_id = self.board.create_bucket(payload)
            self.bytes_sent += len(payload)

            z_mid_np = self._wait_for(bucket_id, session_id, "INFER_RES")
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
