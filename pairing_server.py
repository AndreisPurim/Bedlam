#!/usr/bin/env python3
"""
pairing_server.py — gRPC pairing server for the double-blind mode.

Responsibilities:
 - Generate DH parameters once.
 - Receive client pairing requests (public key only).
 - Receive M2 availability announcements (public key only).
 - When both sides are available, return the counterpart public key and drop
   the queued request/availability. No identities are revealed to peers beyond
   their own IDs and the received public key.

gRPC methods (JSON-in-JSON-out over gRPC):
 - DhParams            {}                           -> {p, g}
 - Request             {client_id, public_key}      -> {assigned, peer_public_key?}
 - PollAssignment      {client_id}                  -> {assigned, peer_public_key?}
 - RegisterM2          {m2_id, public_key}          -> {assigned, peer_public_key?}
 - PollAssignmentM2    {m2_id}                      -> {assigned, peer_public_key?}

We avoid proto generation by using gRPC's generic handlers with JSON
serialization. Payloads are small and only contain base64-encoded public keys.
"""

from __future__ import annotations

import base64
import json
import logging
from collections import deque
from concurrent import futures
import threading

import grpc
from cryptography.hazmat.primitives.asymmetric import dh


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pairing_server")


# ============================================================
# In-memory state + matching
# ============================================================


class PairingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.dh_params = dh.generate_parameters(generator=2, key_size=2048)

        self.waiting_clients: deque[str] = deque()
        self.client_pub: dict[str, bytes] = {}
        self.assign_client: dict[str, bytes] = {}

        self.waiting_m2: deque[str] = deque()
        self.m2_pub: dict[str, bytes] = {}
        self.assign_m2: dict[str, bytes] = {}

    # ---------- core matching ----------
    def handle_client_request(self, client_id: str, public_key: bytes):
        with self.lock:
            self.client_pub[client_id] = public_key
            # If an M2 is waiting, match immediately
            if self.waiting_m2:
                m2_id = self.waiting_m2.popleft()
                m2_pk = self.m2_pub.get(m2_id)
                if m2_pk:
                    self.assign_client[client_id] = m2_pk
                    self.assign_m2[m2_id] = public_key
                    logger.info(
                        f"[pairing] matched client={client_id} with waiting m2={m2_id}; remaining queued clients={len(self.waiting_clients)}"
                    )
                    return m2_pk
            # Otherwise queue the client
            if client_id not in self.waiting_clients:
                self.waiting_clients.append(client_id)
                logger.info(
                    f"[pairing] queued client={client_id}; waiting_clients={len(self.waiting_clients)} waiting_m2={len(self.waiting_m2)}"
                )
            return None

    def handle_m2_register(self, m2_id: str, public_key: bytes):
        with self.lock:
            self.m2_pub[m2_id] = public_key
            # Already assigned? Then don't requeue
            if m2_id in self.assign_m2:
                return self.assign_m2[m2_id]
            # If a client is waiting, match immediately
            if self.waiting_clients:
                client_id = self.waiting_clients.popleft()
                cli_pk = self.client_pub.get(client_id)
                if cli_pk:
                    self.assign_client[client_id] = public_key
                    self.assign_m2[m2_id] = cli_pk
                    logger.info(
                        f"[pairing] matched m2={m2_id} with waiting client={client_id}; remaining waiting_clients={len(self.waiting_clients)}"
                    )
                    return cli_pk
            if m2_id not in self.waiting_m2:
                self.waiting_m2.append(m2_id)
                logger.info(
                    f"[pairing] queued m2={m2_id}; waiting_m2={len(self.waiting_m2)} waiting_clients={len(self.waiting_clients)}"
                )
            return None

    def poll_assignment_client(self, client_id: str):
        with self.lock:
            return self.assign_client.pop(client_id, None)

    def poll_assignment_m2(self, m2_id: str):
        with self.lock:
            return self.assign_m2.pop(m2_id, None)


STATE = PairingState()


# ============================================================
# gRPC helpers (JSON serializers)
# ============================================================


def _serialize(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def _deserialize(data: bytes) -> dict:
    if not data:
        return {}
    return json.loads(data.decode("utf-8"))


def _resp(peer_pk: bytes | None):
    if peer_pk is None:
        return {"assigned": False}
    return {"assigned": True, "peer_public_key": base64.b64encode(peer_pk).decode("ascii")}


# ============================================================
# gRPC method implementations
# ============================================================


def rpc_dh_params(_: dict) -> dict:
    params = STATE.dh_params.parameter_numbers()
    return {"p": params.p, "g": params.g}


def rpc_request(payload: dict) -> dict:
    cid = payload.get("client_id", "")
    pk_b64 = payload.get("public_key", "")
    if not cid or not pk_b64:
        return {"error": "missing client_id/public_key", "assigned": False}
    pk = base64.b64decode(pk_b64)
    logger.info(f"[pairing] client request received cid={cid}")
    peer_pk = STATE.handle_client_request(cid, pk)
    if peer_pk:
        logger.info(f"[pairing] client cid={cid} matched immediately")
    return _resp(peer_pk)


def rpc_poll_assignment(payload: dict) -> dict:
    cid = payload.get("client_id", "")
    if not cid:
        return {"error": "missing client_id", "assigned": False}
    peer_pk = STATE.poll_assignment_client(cid)
    if peer_pk:
        logger.info(f"[pairing] client cid={cid} assignment delivered on poll")
    return _resp(peer_pk)


def rpc_register_m2(payload: dict) -> dict:
    mid = payload.get("m2_id", "")
    pk_b64 = payload.get("public_key", "")
    if not mid or not pk_b64:
        return {"error": "missing m2_id/public_key", "assigned": False}
    pk = base64.b64decode(pk_b64)
    logger.info(f"[pairing] M2 availability registered mid={mid}")
    peer_pk = STATE.handle_m2_register(mid, pk)
    if peer_pk:
        logger.info(f"[pairing] M2 mid={mid} matched immediately")
    return _resp(peer_pk)


def rpc_poll_assignment_m2(payload: dict) -> dict:
    mid = payload.get("m2_id", "")
    if not mid:
        return {"error": "missing m2_id", "assigned": False}
    peer_pk = STATE.poll_assignment_m2(mid)
    if peer_pk:
        logger.info(f"[pairing] M2 mid={mid} assignment delivered on poll")
    return _resp(peer_pk)


# ============================================================
# Server bootstrap
# ============================================================


def serve(host: str = "0.0.0.0", port: int = 50052, max_workers: int = 16):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    generic_handler = grpc.method_handlers_generic_handler(
        "pairing.PairingService",
        {
            "DhParams": grpc.unary_unary_rpc_method_handler(
                lambda req, ctx: rpc_dh_params(req),
                request_deserializer=_deserialize,
                response_serializer=_serialize,
            ),
            "Request": grpc.unary_unary_rpc_method_handler(
                lambda req, ctx: rpc_request(req),
                request_deserializer=_deserialize,
                response_serializer=_serialize,
            ),
            "PollAssignment": grpc.unary_unary_rpc_method_handler(
                lambda req, ctx: rpc_poll_assignment(req),
                request_deserializer=_deserialize,
                response_serializer=_serialize,
            ),
            "RegisterM2": grpc.unary_unary_rpc_method_handler(
                lambda req, ctx: rpc_register_m2(req),
                request_deserializer=_deserialize,
                response_serializer=_serialize,
            ),
            "PollAssignmentM2": grpc.unary_unary_rpc_method_handler(
                lambda req, ctx: rpc_poll_assignment_m2(req),
                request_deserializer=_deserialize,
                response_serializer=_serialize,
            ),
        },
    )
    server.add_generic_rpc_handlers((generic_handler,))

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info(f"Pairing gRPC server listening on {host}:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the double-blind pairing gRPC server")
    parser.add_argument("--host", default="0.0.0.0", help="Host/interface to bind")
    parser.add_argument("--port", type=int, default=50052, help="Port to listen on")
    args = parser.parse_args()
    serve(host=args.host, port=args.port)
