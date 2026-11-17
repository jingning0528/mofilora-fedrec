"""Shared training-log parsing and plotting for the two training figure layouts."""
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def parse_log(path: Path):
    train_turns, train_losses = [], []
    current_turn = None
    pending_eval_turn = None

    # metric_name -> {"turns": [...], "values": [...]}
    eval_metrics = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Train turn
            m_turn = re.search(r"Train Turn\s+(\d+)", line)
            if m_turn:
                current_turn = int(m_turn.group(1))

            # Loss: supports "0.123" and "tensor(0.123)"
            m_loss = re.search(r"Clients average loss:\s*(?:tensor\()?([0-9]*\.?[0-9]+)", line)
            if m_loss and current_turn is not None:
                train_turns.append(current_turn)
                train_losses.append(float(m_loss.group(1)))

            # Eval turn
            m_eval = re.search(r"Eval @ Turn\s+(\d+)", line)
            if m_eval:
                pending_eval_turn = int(m_eval.group(1))

            # Metrics line (generic parser)
            if "[Metrics]" in line:
                turn = pending_eval_turn if pending_eval_turn is not None else current_turn
                if turn is not None:
                    # parse pairs like "NDCG(10): 0.088666" or "MRR: 0.083082"
                    pairs = re.findall(r"([A-Za-z0-9_()]+)\s*:\s*([0-9]*\.?[0-9]+)", line)
                    for metric_name, metric_val in pairs:
                        if metric_name not in eval_metrics:
                            eval_metrics[metric_name] = {"turns": [], "values": []}
                        eval_metrics[metric_name]["turns"].append(turn)
                        eval_metrics[metric_name]["values"].append(float(metric_val))
                pending_eval_turn = None

    # convert arrays
    train_turns = np.array(train_turns, dtype=int)
    train_losses = np.array(train_losses, dtype=float)

    for k in eval_metrics:
        turns = np.array(eval_metrics[k]["turns"], dtype=int)
        vals = np.array(eval_metrics[k]["values"], dtype=float)
        n = min(len(turns), len(vals))
        eval_metrics[k]["turns"] = turns[:n]
        eval_metrics[k]["values"] = vals[:n]

    return {
        "train_turns": train_turns,
        "train_losses": train_losses,
        "eval_metrics": eval_metrics,
    }


def smooth(y, w=9):
    if len(y) == 0:
        return np.array([]), 0
    w = max(1, min(w, len(y)))
    kernel = np.ones(w) / w
    ys = np.convolve(y, kernel, mode="valid")
    return ys, w // 2


def sample_every_n_rounds(x, y, n=10):
    if len(x) == 0 or len(y) == 0:
        return np.array([]), np.array([])
    m = min(len(x), len(y))
    x = np.asarray(x[:m])
    y = np.asarray(y[:m])
    mask = (x % n == 0)
    return x[mask], y[mask]


def plot_metric(data, x_key, y_key, title, y_label, out_path):
    plt.figure(figsize=(4.5, 3.5))

    for name, d in data.items():
        x = d.get(x_key, np.array([]))
        y = d.get(y_key, np.array([]))

        if len(x) == 0 or len(y) == 0:
            continue

        m = min(len(x), len(y))

        plt.plot(
            x[:m],
            y[:m],
            linewidth=2.5,
            label=name,
        )

    plt.xlabel("Communication Round", fontsize=16)
    plt.ylabel(y_label, fontsize=16)

    plt.tick_params(axis="both", labelsize=14)

    plt.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )

    plt.legend(
        fontsize=13,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
