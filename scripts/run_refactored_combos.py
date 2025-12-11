#!/usr/bin/env python3
"""
Lightweight runner to exercise all refactored architectures:
  - vanilla-split
  - board-blind
  - single-blind-bucket
  - double-blind

For each combination of peer count, model, and pad size, this script:
  - writes a temporary config.yaml,
  - starts board (and pairing for double-blind),
  - runs client.py,
  - tears down background processes,
  - restores the original config at the end.

Adjust the PEER_COUNTS/MODELS/PAD_MULTIPLES/ARCHITECTURES lists as needed.
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
from typing import Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
LOG_DIR = REPO_ROOT / "runs" / "automation_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Tunables
PEER_COUNTS = [3, 4, 5, 6, 7]
MODELS = ["default"]
PAD_MULTIPLES = [1024]
ARCHITECTURES = ["vanilla-split", "board-blind", "single-blind-bucket", "double-blind"]
BOARD_HOST = "127.0.0.1"
BOARD_PORT = 50051
PAIRING_HOST = "127.0.0.1"
PAIRING_PORTS = [50052, 50062, 50072]


def load_base_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def write_config(cfg: dict, path: Path):
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def build_peers(num_m1m3: int, num_m2: int) -> Dict[str, List[dict]]:
    m2_peers = [{"name": f"m2_{i + 1}", "key": f"secret_key_{i + 1}"} for i in range(num_m2)]
    m1m3_peers = [{"name": f"client_{i + 1}", "target_m2": m2_peers[i % num_m2]["name"]} for i in range(num_m1m3)]
    return {"M2": m2_peers, "M1M3": m1m3_peers}


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def ensure_pairing_server(host: str, port: int, run_name: str):
    """
    Start pairing server if not already reachable. Returns (proc, already_running: bool).
    """
    if wait_for_port(host, port, timeout=1.0):
        print(f"[info] Pairing server already running for {run_name} on {host}:{port}", flush=True)
        return None, True

    pairing_proc = start_process(
        [sys.executable, "pairing_server.py", "--host", host, "--port", str(port)],
        log_file=LOG_DIR / f"{run_name}_pairing.log",
    )
    if not wait_for_port(host, port, timeout=10):
        # Retry once
        stop_process(pairing_proc)
        pairing_proc = start_process(
            [sys.executable, "pairing_server.py", "--host", host, "--port", str(port)],
            log_file=LOG_DIR / f"{run_name}_pairing_retry.log",
        )
        if not wait_for_port(host, port, timeout=10):
            stop_process(pairing_proc)
            return None, False
    print(f"[info] Pairing server ready for {run_name} on {host}:{port}", flush=True)
    return pairing_proc, False


def start_process(cmd: List[str], log_file: Path | None = None, env: dict | None = None) -> subprocess.Popen:
    stdout = stderr = subprocess.DEVNULL
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        stdout = log_file.open("wb")
        stderr = subprocess.STDOUT
    return subprocess.Popen(cmd, stdout=stdout, stderr=stderr, cwd=REPO_ROOT, env=env)


def stop_process(proc: subprocess.Popen | None, timeout: float = 5.0):
    if proc is None or proc.poll() is not None:
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
    pad: int,
    num_m1m3: int,
    num_m2: int,
    pairing_port: int,
) -> dict:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("run", {})["name"] = run_name

    cfg.setdefault("general", {})
    cfg["general"]["architecture"] = arch
    cfg["general"]["model_architecture"] = model
    cfg["general"]["pad_multiple"] = pad
    cfg["general"]["board_host"] = BOARD_HOST
    cfg["general"]["board_port"] = BOARD_PORT
    cfg["general"]["pairing_host"] = PAIRING_HOST
    cfg["general"]["pairing_port"] = pairing_port

    cfg["peers"] = build_peers(num_m1m3, num_m2)
    cfg.setdefault("double_blind", {})
    cfg["double_blind"]["m1m3_count"] = num_m1m3
    cfg["double_blind"]["m2_count"] = num_m2
    return cfg


def run_combo(
    arch: str,
    model: str,
    pad: int,
    num_m1m3: int,
    num_m2: int,
    base_cfg: dict,
    dry_run: bool,
    timeout: int,
    pairing_port: int,
):
    suffix = f"p{num_m1m3}m2_{num_m2}" if num_m2 != num_m1m3 else f"p{num_m1m3}"
    run_name = f"auto_{arch}_{suffix}_{model}_pad{pad}_{int(time.time())}"
    cfg = prepare_config(base_cfg, run_name, arch, model, pad, num_m1m3, num_m2, pairing_port)

    board_proc = pairing_proc = ray_proc = None
    combo_log = LOG_DIR / f"{run_name}.log"
    ray_stop()

    try:
        env = os.environ.copy()
        env["RAY_DISABLE_USAGE_STATS"] = "1"

        # Start board for all modes (vanilla uses plaintext over the same gRPC service)
        board_proc = start_process(
            [sys.executable, "board_server.py", "--host", BOARD_HOST, "--port", str(BOARD_PORT)],
            log_file=LOG_DIR / f"{run_name}_board.log",
        )
        if not wait_for_port(BOARD_HOST, BOARD_PORT, timeout=10):
            print(f"[warn] Board server not reachable for {run_name}; skipping")
            return

        # Pairing server for double-blind
        pairing_already = False
        if arch == "double-blind":
            pairing_proc, pairing_already = ensure_pairing_server(PAIRING_HOST, pairing_port, run_name)
            if pairing_proc is None and pairing_already is False:
                print(f"[warn] Pairing server not reachable for {run_name}; skipping")
                return

        # Persist config
        config_path = REPO_ROOT / "config.yaml"
        write_config(cfg, config_path)

        if dry_run:
            print(f"[dry-run] {run_name}")
            return

        entrypoint = [sys.executable, "client.py", str(config_path)]
        print(f"[run] {run_name} m1m3={num_m1m3} m2={num_m2} arch={arch} model={model} pad={pad}", flush=True)
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

        if combo_proc.returncode not in (0, None):
            print(f"[warn] Run exited with code {combo_proc.returncode} for {run_name} (see log {combo_log})")
        else:
            print(f"[done] {run_name} completed", flush=True)
    finally:
        stop_process(board_proc)
        if not pairing_already:
            stop_process(pairing_proc)
        stop_process(ray_proc)
        ray_stop()


def main(args):
    base_cfg = load_base_config(DEFAULT_CONFIG)
    backup_path = LOG_DIR / "config_backup.yaml"
    write_config(base_cfg, backup_path)

    pairing_index = 0

    try:
        for num_m1m3 in PEER_COUNTS:
            combos = [(num_m1m3, num_m1m3), (num_m1m3, max(1, num_m1m3 // 2)), (num_m1m3, num_m1m3 * 2)]
            for model in MODELS:
                for pad in PAD_MULTIPLES:
                    for arch in ARCHITECTURES:
                        for m1m3_count, m2_count in combos:
                            pairing_port = PAIRING_PORTS[pairing_index % len(PAIRING_PORTS)]
                            pairing_index += 1
                            run_combo(
                                arch=arch,
                                model=model,
                                pad=pad,
                                num_m1m3=m1m3_count,
                                num_m2=m2_count,
                                base_cfg=base_cfg,
                                dry_run=args.dry_run,
                                timeout=args.timeout,
                                pairing_port=pairing_port,
                            )
    finally:
        write_config(base_cfg, DEFAULT_CONFIG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run refactored OMRsplit architectures.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Optional per-run timeout in seconds (0 means wait indefinitely).",
    )
    args = parser.parse_args()
    main(args)
