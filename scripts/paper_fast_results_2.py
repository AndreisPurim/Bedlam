#!/usr/bin/env python3
"""
Fast paper results (Option 2):
- vanilla-split (mnist)
- double-blind (mnist) WITHOUT PIR

Use this to get double-blind plots without the PIR path.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
BOARD_HOST = "127.0.0.1"
BOARD_PORT = 50051
PAIRING_HOST = "127.0.0.1"


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


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


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
    dataset: str,
    model: str,
    pad: int,
    num_m1m3: int,
    num_m2: int,
    epochs: int,
    parent_run_dir: Path,
    pairing_port: int,
) -> dict:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("run", {})["name"] = run_name
    cfg["run"]["base_dir"] = str(parent_run_dir)

    cfg.setdefault("general", {})
    cfg["general"]["architecture"] = arch
    cfg["general"]["model_architecture"] = model
    cfg["general"]["dataset"] = dataset
    cfg["general"]["epochs"] = epochs
    cfg["general"]["pad_multiple"] = pad
    cfg["general"]["board_host"] = BOARD_HOST
    cfg["general"]["board_port"] = BOARD_PORT
    cfg["general"]["pairing_host"] = PAIRING_HOST
    cfg["general"]["pairing_port"] = int(pairing_port)
    cfg["general"]["enable_perf_metrics"] = True
    cfg["general"]["use_pir"] = False

    cfg["peers"] = build_peers(num_m1m3, num_m2)
    cfg.setdefault("double_blind", {})
    cfg["double_blind"]["m1m3_count"] = num_m1m3
    cfg["double_blind"]["m2_count"] = num_m2
    return cfg


def run_one(
    base_cfg: dict,
    parent_run_dir: Path,
    log_dir: Path,
    arch: str,
    dataset: str,
    model: str,
    pad: int,
    num_m1m3: int,
    num_m2: int,
    epochs: int,
    timeout: int,
    enable_gpu: bool,
):
    run_name = f"paper_fast_{arch}_{dataset}_{model}_p{num_m1m3}m2_{num_m2}_{int(time.time())}"
    run_dir = parent_run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pairing_port = find_free_port(PAIRING_HOST) if arch == "double-blind" else 0
    cfg = prepare_config(
        base_cfg,
        run_name,
        arch,
        dataset,
        model,
        pad,
        num_m1m3,
        num_m2,
        epochs,
        parent_run_dir,
        pairing_port,
    )

    board_proc = None
    pairing_proc = None
    combo_log = log_dir / f"{run_name}.log"
    ray_stop()

    try:
        env = os.environ.copy()
        env["RAY_DISABLE_USAGE_STATS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "0" if enable_gpu else ""

        board_proc = start_process(
            [
                sys.executable,
                "board_server.py",
                "--host",
                BOARD_HOST,
                "--port",
                str(BOARD_PORT),
                "--metrics-path",
                str(run_dir / "board_metrics.csv"),
            ],
            log_file=log_dir / f"{run_name}_board.log",
            env=env,
        )
        if not wait_for_port(BOARD_HOST, BOARD_PORT, timeout=10):
            print(f"[warn] Board server not reachable for {run_name}; skipping")
            return

        if arch == "double-blind":
            pairing_log = log_dir / f"{run_name}_pairing.log"
            pairing_proc = start_process(
                [sys.executable, "pairing_server.py", "--host", PAIRING_HOST, "--port", str(pairing_port)],
                log_file=pairing_log,
                env=env,
            )
            if not wait_for_port(PAIRING_HOST, pairing_port, timeout=10):
                print(f"[warn] Pairing server not reachable for {run_name}; skipping")
                return

        write_config(cfg, DEFAULT_CONFIG)
        entrypoint = [sys.executable, "client.py", str(DEFAULT_CONFIG)]
        print(f"[run] {run_name} arch={arch} dataset={dataset} model={model}", flush=True)
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
            print(f"[warn] Run exited with code {combo_proc.returncode} for {run_name}")
        else:
            print(f"[done] {run_name} completed", flush=True)
            plots_dir = parent_run_dir / "plots"
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "scripts" / "plot_paper_figures.py"),
                        "--runs-glob",
                        str(parent_run_dir / "*"),
                        "--out-dir",
                        str(plots_dir),
                    ],
                    cwd=REPO_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"[plots] {plots_dir}", flush=True)
            except Exception:
                print(f"[warn] plot generation failed for {run_name}", flush=True)
    finally:
        stop_process(board_proc)
        stop_process(pairing_proc)
        ray_stop()


def main(args):
    base_cfg = load_base_config(DEFAULT_CONFIG)

    parent_name = args.paper_name or f"paper_fast_run_{int(time.time())}"
    parent_run_dir = REPO_ROOT / "runs" / parent_name
    parent_run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = parent_run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("vanilla-split", "mnist"),
        ("double-blind", "mnist"),
    ]
    for arch, dataset in runs:
        run_one(
            base_cfg=base_cfg,
            parent_run_dir=parent_run_dir,
            log_dir=log_dir,
            arch=arch,
            dataset=dataset,
            model=args.model,
            pad=args.pad,
            num_m1m3=args.clients,
            num_m2=args.m2,
            epochs=args.epochs,
            timeout=args.timeout,
            enable_gpu=args.enable_gpu,
        )
    print(f"[paper-fast-2] {parent_run_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast paper runs (option 2).")
    parser.add_argument("--clients", type=int, default=1, help="Number of M1M3 clients.")
    parser.add_argument("--m2", type=int, default=1, help="Number of M2 peers.")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per run.")
    parser.add_argument("--model", type=str, default="default", help="Model architecture.")
    parser.add_argument("--pad", type=int, default=1024, help="Pad multiple.")
    parser.add_argument("--timeout", type=int, default=0, help="Per-run timeout (seconds).")
    parser.add_argument("--enable-gpu", action="store_true", help="Allow GPU usage during runs.")
    parser.add_argument("--paper-name", type=str, default="", help="Parent folder name under runs/ (optional).")
    args = parser.parse_args()
    main(args)
