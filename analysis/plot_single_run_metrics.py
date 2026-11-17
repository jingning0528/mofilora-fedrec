import re
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt



if __package__:
    from ._training_plot_utils import parse_log, smooth, sample_every_n_rounds, plot_metric
else:
    from _training_plot_utils import parse_log, smooth, sample_every_n_rounds, plot_metric

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "log"
FIG_DIR = REPO_ROOT / "figures"

# ===== one input file only =====
INPUT_LOG_FILE = "baseline_1000/LoRA.txt"

# metrics
HR_METRIC = "HR(10)"
NDCG_METRIC = "NDCG(10)"

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})










def main():
    log_path = LOG_DIR / INPUT_LOG_FILE
    if not log_path.exists():
        raise FileNotFoundError(f"File not found: {log_path}")

    d = parse_log(log_path)

    # Loss (smoothed + sampled)
    ys, off = smooth(d["train_losses"], w=9)
    xs = d["train_turns"][off:off + len(ys)] if len(ys) > 0 else np.array([])
    loss_x, loss_y = sample_every_n_rounds(xs, ys, n=10)

    # HR / NDCG
    hr_x = hr_y = np.array([])
    ndcg_x = ndcg_y = np.array([])

    if HR_METRIC in d["eval_metrics"]:
        hr_x = d["eval_metrics"][HR_METRIC]["turns"]
        hr_y = d["eval_metrics"][HR_METRIC]["values"]
    else:
        print(f"[Warning] {HR_METRIC} not found in {log_path.name}")

    if NDCG_METRIC in d["eval_metrics"]:
        ndcg_x = d["eval_metrics"][NDCG_METRIC]["turns"]
        ndcg_y = d["eval_metrics"][NDCG_METRIC]["values"]
    else:
        print(f"[Warning] {NDCG_METRIC} not found in {log_path.name}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_loss = FIG_DIR / f"{ts}_{log_path.stem}_loss.pdf"
    out_hr_ndcg = FIG_DIR / f"{ts}_{log_path.stem}_hr_ndcg.pdf"

    # Figure 1: Loss
    loss_data = {
        "Loss": {
            "x": loss_x,
            "y": loss_y,
        }
    }
    plot_metric(
        data=loss_data,
        x_key="x",
        y_key="y",
        title="Loss",
        y_label="Loss",
        out_path=out_loss,
    )

    # Figure 2: HR + NDCG
    metric_data = {
        HR_METRIC: {
            "x": hr_x,
            "y": hr_y,
        },
        NDCG_METRIC: {
            "x": ndcg_x,
            "y": ndcg_y,
        },
    }
    plot_metric(
        data=metric_data,
        x_key="x",
        y_key="y",
        title="HR and NDCG",
        y_label="Value",
        out_path=out_hr_ndcg,
    )

    print(f"Saved: {out_loss}")
    print(f"Saved: {out_hr_ndcg}")


if __name__ == "__main__":
    main()