#!/usr/bin/env python3
import argparse
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(run_dir: str):
    pattern = os.path.join(run_dir, "metrics_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No metrics_*.csv files found in {run_dir}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        # columns: epoch,step,loss,acc
        df["client"] = os.path.splitext(os.path.basename(fp))[0].replace("metrics_", "")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    # global step per client
    data["client_step"] = data.groupby("client").cumcount()
    return data, files


def main():
    parser = argparse.ArgumentParser(description="Plot Split Learning metrics from metrics_*.csv")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory (e.g., runs/run_...)")
    parser.add_argument("--out", type=str, default="results.png", help="Output PNG path")
    parser.add_argument("--per-client", action="store_false", help="Draw per-client curves")
    args = parser.parse_args()

    data, files = load_metrics(args.run_dir)
    print(f"Loaded {len(files)} metric file(s):")
    for f in files:
        print(f" - {f}")

    # aggregate by client_step
    agg = (
        data.groupby("client_step")
        .agg(loss=("loss", "mean"), acc=("acc", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    if args.per_client:
        for name, g in data.groupby("client"):
            ax[0].plot(g["client_step"], g["loss"], alpha=0.3, linewidth=1, label=f"{name}")
    ax[0].plot(agg["client_step"], agg["loss"], linewidth=2, label="mean")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Client step")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True, linestyle=":")
    if args.per_client:
        ax[0].legend()

    # Accuracy
    if args.per_client:
        for name, g in data.groupby("client"):
            ax[1].plot(g["client_step"], g["acc"], alpha=0.3, linewidth=1, label=f"{name}")
    ax[1].plot(agg["client_step"], agg["acc"], linewidth=2, label="mean")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Client step")
    ax[1].set_ylabel("Acc")
    ax[1].grid(True, linestyle=":")
    if args.per_client:
        ax[1].legend()

    fig.suptitle("Split Learning — Training Metrics")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
