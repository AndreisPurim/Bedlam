#!/usr/bin/env python3
"""
Exhaustively run combinations of peer counts, model architectures, pad sizes,
and system modes for this repo.

Rules implemented from the request:
- Peer counts: M1M3 and M2 are equal, varying 1..10.
- Model architectures: default, mlp, deep, resnet-lite.
- pad_multiple: 1024, 2048.
- Modes: vanilla-split, board-blind, single-blind-two-pools, single-blind-bucket, double-blind.
- For single-blind-two-pools, estimate Ray memory per combo; set it for the run,
  and skip if the estimate exceeds 3GB.
- For double-blind, ensure the pairing server is reachable before running.
- After every combo, kill any servers and Ray processes that were started.

This script edits a temporary config per combo, starts the necessary servers,
runs the appropriate entrypoint, and tears everything down between runs.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
LOG_DIR = REPO_ROOT / "runs" / "automation_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# PEER_COUNTS = list(range(1, 11))
PEER_COUNTS = [2]
MODELS = ["default", "mlp", "deep", "resnet-lite"]
PAD_MULTIPLES = [1024]
ARCHITECTURES = [
    "vanilla-split",
    "board-blind",
    "single-blind-bucket",
    "single-blind-two-pools",
    # "double-blind",
]

SINGLE_BLIND_MAX_GB = 3.0
BOARD_HOST = "127.0.0.1"
BOARD_PORT = 50051
PAIRING_HOST = "127.0.0.1"
PAIRING_PORT = 50052


def load_base_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def build_peers(num_peers: int) -> Dict[str, List[dict]]:
    m2_peers = [{"name": f"m2_{i + 1}", "key": f"secret_key_{i + 1}"} for i in range(num_peers)]
    m1m3_peers = []
    for i in range(num_peers):
        target = m2_peers[i % num_peers]["name"]
        m1m3_peers.append({"name": f"client_{i + 1}", "target_m2": target})
    return {"M2": m2_peers, "M1M3": m1m3_peers}


def estimate_single_blind_memory_gb(num_peers: int) -> Tuple[float, float]:
    """Return (total_gb, object_store_gb). Conservative to avoid crashes."""
    workers = num_peers * 2
    per_worker_gb = 0.35  # TensorFlow runtime per Ray worker
    object_store_gb = 1.0  # fits MNIST shards with headroom
    overhead_gb = 0.3      # logging + protocol buffers
    total_gb = workers * per_worker_gb + object_store_gb + overhead_gb
    return total_gb, object_store_gb


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def start_process(cmd: List[str], log_file: Path | None = None, env: dict | None = None) -> subprocess.Popen:
    stdout = stderr = subprocess.DEVNULL
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        stdout = log_file.open("wb")
        stderr = subprocess.STDOUT
    return subprocess.Popen(cmd, stdout=stdout, stderr=stderr, cwd=REPO_ROOT, env=env)


def stop_process(proc: subprocess.Popen | None, name: str, timeout: float = 5.0):
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
    except Exception:
        pass


def ray_stop():
    subprocess.run(["ray", "stop", "--force"], cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prepare_config(
    base_cfg: dict,
    run_name: str,
    arch: str,
    model: str,
    pad_multiple: int,
    num_peers: int,
) -> dict:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("run", {})
    cfg["run"]["name"] = run_name

    cfg.setdefault("general", {})
    cfg["general"]["architecture"] = arch
    cfg["general"]["model_architecture"] = model
    cfg["general"]["pad_multiple"] = pad_multiple
    cfg["general"]["board_host"] = BOARD_HOST
    cfg["general"]["board_port"] = BOARD_PORT
    cfg["general"]["pairing_host"] = PAIRING_HOST
    cfg["general"]["pairing_port"] = PAIRING_PORT

    cfg["peers"] = build_peers(num_peers)

    cfg.setdefault("double_blind", {})
    cfg["double_blind"]["m1m3_count"] = num_peers
    cfg["double_blind"]["m2_count"] = num_peers
    return cfg


def run_combo(
    arch: str,
    model: str,
    pad: int,
    num_peers: int,
    base_cfg: dict,
    dry_run: bool = False,
    timeout: int = 0,
):
    run_name = f"auto_{arch}_p{num_peers}_{model}_pad{pad}_{int(time.time())}"
    cfg = prepare_config(base_cfg, run_name, arch, model, pad, num_peers)

    board_proc = None
    pairing_proc = None
    ray_proc = None

    combo_log = LOG_DIR / f"{run_name}.log"
    ray_stop()

    try:
        # Single-blind memory handling
        env = os.environ.copy()
        object_store_bytes = None
        if arch == "single-blind-two-pools":
            total_gb, obj_gb = estimate_single_blind_memory_gb(num_peers)
            if total_gb > SINGLE_BLIND_MAX_GB:
                print(f"[skip] {run_name}: estimated {total_gb:.1f}GB exceeds {SINGLE_BLIND_MAX_GB}GB cap")
                return
            object_store_bytes = int(obj_gb * 1024 ** 3)
            total_bytes = int(total_gb * 1024 ** 3)
            ray_cmd = [
                "ray",
                "start",
                "--head",
                f"--object-store-memory={object_store_bytes}",
                f"--memory={total_bytes}",
                "--disable-usage-stats",
            ]
            ray_proc = start_process(ray_cmd, log_file=LOG_DIR / f"{run_name}_ray_head.log")
            if ray_proc.wait(timeout=20) != 0:
                print(f"[warn] Ray head failed to start for {run_name}")
            env["RAY_ADDRESS"] = "auto"
            env["RAY_DISABLE_USAGE_STATS"] = "1"

        # Servers
        if arch != "vanilla-split":
            board_proc = start_process(
                [sys.executable, "board_server.py", "--host", BOARD_HOST, "--port", str(BOARD_PORT)],
                log_file=LOG_DIR / f"{run_name}_board.log",
            )
            if not wait_for_port(BOARD_HOST, BOARD_PORT, timeout=10):
                print(f"[warn] Board server not reachable for {run_name}; skipping run")
                return

        if arch == "double-blind":
            pairing_proc = start_process(
                [sys.executable, "pairing_server.py", "--host", PAIRING_HOST, "--port", str(PAIRING_PORT)],
                log_file=LOG_DIR / f"{run_name}_pairing.log",
            )
            if not wait_for_port(PAIRING_HOST, PAIRING_PORT, timeout=10):
                print(f"[warn] Pairing server not reachable for {run_name}; skipping run")
                return

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp)
            cfg_path = tmp.name

        if dry_run:
            print(f"[dry-run] Would run {run_name} with config {cfg_path}")
            return

        entrypoint = [sys.executable, "client.py", cfg_path]
        combo_proc = start_process(entrypoint, log_file=combo_log, env=env)
        if timeout:
            try:
                combo_proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                combo_proc.send_signal(signal.SIGINT)
                try:
                    combo_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    combo_proc.kill()
        else:
            combo_proc.wait()
    finally:
        stop_process(board_proc, "board")
        stop_process(pairing_proc, "pairing")
        stop_process(ray_proc, "ray-head")
        ray_stop()


def main(args):
    base_cfg = load_base_config(DEFAULT_CONFIG)
    print("combinations: ", len(PEER_COUNTS) * len(MODELS) * len(PAD_MULTIPLES) * len(ARCHITECTURES))
    for num_peers in PEER_COUNTS:
        for model in MODELS:
            for pad in PAD_MULTIPLES:
                for arch in ARCHITECTURES:
                    run_combo(
                        arch=arch,
                        model=model,
                        pad=pad,
                        num_peers=num_peers,
                        base_cfg=base_cfg,
                        dry_run=args.dry_run,
                        timeout=args.timeout,
                    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all OMRsplit experiment combinations.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Optional per-run timeout in seconds (0 means wait indefinitely).",
    )
    args = parser.parse_args()
    main(args)
