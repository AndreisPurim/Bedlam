import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


HOST = "127.0.0.1"
PORT = int(os.environ.get("BOARD_SMOKE_PORT", "50056"))


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
    repo_root = Path(__file__).resolve().parents[1]
    board_proc = subprocess.Popen(
        [sys.executable, "board_server.py", "--host", HOST, "--port", str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        cwd=repo_root,
    )
    try:
        if not wait_for_port(HOST, PORT, timeout=10):
            raise RuntimeError("board_server did not start")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_name = f"smoke_federated_{int(time.time())}"
            cfg = {
                "run": {"base_dir": str(Path(tmpdir) / "runs"), "name": run_name},
                "general": {
                    "architecture": "federated",
                    "dataset": "mnist",
                    "model_architecture": "default",
                    "batch_size": 256,
                    "lr": 0.001,
                    "log_level": "INFO",
                    "suppress_warnings": True,
                    "board_host": HOST,
                    "board_port": PORT,
                    "random_seed": 42,
                },
                "federated": {
                    "rounds": 1,
                    "local_epochs": 1,
                    "server_name": "fed_server",
                },
                "peers": {
                    "M1M3": [
                        {"name": "client_1", "target_m2": "m2_1"},
                        {"name": "client_2", "target_m2": "m2_1"},
                    ],
                    "M2": [{"name": "m2_1", "key": "secret_key_1"}],
                },
                "double_blind": {"m1m3_count": 0, "m2_count": 0},
            }

            config_path = Path(tmpdir) / "config.yaml"
            with config_path.open("w") as f:
                yaml.safe_dump(cfg, f)

            proc = subprocess.Popen(
                [sys.executable, "client.py", str(config_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=repo_root,
            )
            try:
                proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RuntimeError("federated smoke timed out")
            if proc.returncode not in (0, None):
                raise RuntimeError(f"federated smoke failed with code {proc.returncode}")

        print("smoke ok")
    finally:
        board_proc.send_signal(signal.SIGINT)
        try:
            board_proc.wait(timeout=5)
        except Exception:
            board_proc.kill()


if __name__ == "__main__":
    main()
