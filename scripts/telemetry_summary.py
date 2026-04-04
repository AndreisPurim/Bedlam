#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

BYTES_RE = re.compile(r"\[bytes-summary\] sent=(\d+)B recv=(\d+)B")


def parse_metrics(path: Path):
    try:
        with path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        last = rows[-1]
        return {
            "loss": float(last.get("loss", 0.0)),
            "acc": float(last.get("acc", 0.0)),
        }
    except Exception:
        return None


def parse_bytes_summary(path: Path):
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None
    matches = BYTES_RE.findall(text)
    if not matches:
        return None
    sent, recv = matches[-1]
    return {"sent": int(sent), "recv": int(recv)}


def summarize_run(run_dir: Path):
    metrics = list(run_dir.glob("metrics_*.csv"))
    m2_logs = list(run_dir.glob("m2_*.log")) + list(run_dir.glob("db_m2_*.log"))
    client_logs = list(run_dir.glob("client_*.log")) + list(run_dir.glob("db_client_*.log"))

    metrics_vals = [parse_metrics(p) for p in metrics]
    metrics_vals = [m for m in metrics_vals if m]
    mean_acc = sum(m["acc"] for m in metrics_vals) / len(metrics_vals) if metrics_vals else 0.0
    mean_loss = sum(m["loss"] for m in metrics_vals) / len(metrics_vals) if metrics_vals else 0.0

    total_sent = 0
    total_recv = 0
    for log in client_logs + m2_logs:
        b = parse_bytes_summary(log)
        if b:
            total_sent += b["sent"]
            total_recv += b["recv"]

    return {
        "run": run_dir.name,
        "clients": len(metrics),
        "m2_peers": len(m2_logs),
        "mean_acc": round(mean_acc, 6),
        "mean_loss": round(mean_loss, 6),
        "total_sent": total_sent,
        "total_recv": total_recv,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize run telemetry.")
    parser.add_argument("--run_dir", type=str, default="", help="Specific run directory.")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Parent runs directory.")
    parser.add_argument("--output", type=str, default="", help="Optional CSV output path.")
    parser.add_argument("--append", action="store_true", help="Append to CSV if it exists.")
    args = parser.parse_args()

    run_dirs = []
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        runs_root = Path(args.runs_dir)
        run_dirs = [
            p
            for p in runs_root.iterdir()
            if p.is_dir() and (p.name.startswith("run_") or p.name.startswith("auto_"))
        ]

    summaries = [summarize_run(p) for p in run_dirs]
    summaries = [s for s in summaries if s]
    if not summaries:
        print("No runs found.")
        return

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append and out_path.exists() else "w"
        with out_path.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
            if mode == "w":
                writer.writeheader()
            for row in summaries:
                writer.writerow(row)

    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
