import os
import re
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


#
if __package__:
    from ._training_plot_utils import parse_log, smooth, sample_every_n_rounds, plot_metric
else:
    from _training_plot_utils import parse_log, smooth, sample_every_n_rounds, plot_metric

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "log"
FIG_DIR = REPO_ROOT / "figures"

# Set log file names here (1 to 10 files)
SELECTED_LOG_FILES = [

    #########################################Instrument########################################
    "Instrument/Full.txt",
    "Instrument/LoRA.txt",
    "Instrument/LoRA-FixedB.txt",
    "Instrument/MoFiLoRA.txt",

    # "instrument/LoRA.txt",
    # "instrument/MoFiLoRA.txt",
    # "software_1800/MoFiLoRA_1800.txt",
    # "multi_seed/Instrument/Instrument_Full_2024.txt",
    # "multi_seed/Instrument/Instrument_LoRA_2024.txt",
    # "multi_seed/Instrument/Instrument_LoRA_FixedA_2024.txt",
    # "multi_seed/Instrument/Instrument_LoRA_FixedB_2024.txt",
    # "multi_seed/Instrument/Instrument_LoRA_MomA_2024.txt",
    # "multi_seed/Instrument/Instrument_LoRA_MomB_2024.txt",
    # "multi_seed/Instrument/Instrument_LoRA_MomB_FixedA_2024.txt",
    # "multi_seed/Instrument/Instrument_LoRA_MomA_FixedB_2024.txt",


    #########################################DoRA########################################
    # "DoRA/DoRA.txt",

    #########################################Paper########################################
    # "software/Full.txt",
    # "software/LoRA.txt",
    # "software/LoRA-FixedB.txt",
    # "software/MofiLoRA.txt",

    # "industrial/Full.txt",
    # "industrial/LoRA.txt",
    # "industrial/LoRA-FixedB.txt",
    # "industrial/MofiLoRA.txt",
    # "industrial/MofiLoRA_2.txt",
    # "industrial/MoFiLoRA-0.9-1.0.txt",
    # "industrial/MoFiLoRA-0.8-1.0.txt",

    # "baseline_2000/Full.txt",
    # "baseline_2000/LoRA.txt",
    # "baseline_2000/LoRA-FixedB.txt",
    # "baseline_2000/MoFiLoRA.txt",

    # "baseline_1000/Full.txt",
    # "baseline_1000/LoRA.txt",
    # "AB_test_1000/LoRA-FixedB.txt",
    # "baseline_1000/MoFiLoRA.txt",

    # "baseline_1000/Full.txt",
    # "baseline_1000/LoRA.txt",
    # "AB_test_1000/LoRA-FixedA.txt",
    # "AB_test_1000/LoRA-FixedB.txt",

    # "AB_test_1000/LoRA-MomA.txt",
    # "AB_test_1000/LoRA-MomB.txt",
    # "AB_test_1000/LoRA-MomB-FixedA.txt",
    # "AB_test_1000/LoRA-MomA-FixedB.txt",

    # "software/Full.txt",
    # "software/LoRA.txt",
    # "software/MoFiLoRA.txt",
    # "software/MoFiLoRA-0.99-2.0.txt",
    # "software/MoFiLoRA-0.9-2.0.txt",
    # "software/MoFiLoRA-0.9-1.0.txt",
    # "software/MoFiLoRA-0.8-0.5.txt",
    # "software/MoFiLoRA-0.99-1.0.txt",
    # "software/MoFiLoRA-0.8-2.0.txt",

    # "industrial/FedNCF-Full.txt",
    # "industrial/FedNCF-LoRA.txt",
    # "industrial/MoFiLoRA.txt",
    # "industrial/MoFiLoRA-0.9-1.0.txt",
    # "industrial/MoFiLoRA-0.9-0.5.txt",
    # "industrial/MoFiLoRA-0.8-1.0.txt",


    # **************************************baseline********************************
    # "baseline_1000/FedNCF-Full.txt",
    # "baseline_1000/FedNCF-LoRA.txt",
    # "baseline_1000/MofiLoRA.txt",
    # "baseline_1000/FedNCF-Full-Mom.txt",

    # "baseline_2000/FedNCF-Full.txt",
    # "baseline_2000/FedNCF-LoRA.txt",
    # "baseline_2000/FedNCF-FixedB.txt",
    # "baseline_2000/MofiLoRA.txt",

    # "baseline_500/fedncf_lora.txt",
    # "baseline_500/fedncf_base.txt",

    # "baseline_200/fedncf_lora.txt",
    # "baseline_200/fedncf_base.txt",

    # **************************************Seed*********************************************
    # "seed/MofiLoRA-2024.txt",
    # "seed/MofiLoRA-2025.txt",
    # "seed/MofiLoRA-2026.txt",
    # "seed/MofiLoRA-2027.txt",
    # "seed/MofiLoRA-2028.txt",

    # **************************************EMA Finetune 1000********************************
    # "finetune_1000/ema/ema_0.97_5.0.txt",
    # "finetune_1000/ema/ema_0.97_10.0.txt",
    # "finetune_1000/ema/ema_0.97_20.0.txt",

    # "finetune_1000/ema/ema_0.8_10.0.txt",
    # "finetune_1000/ema/ema_0.97_10.0.txt",
    # "finetune_1000/ema/ema_0.99_10.0.txt",

    # "finetune_1000/ema/ema_0.8_5.0.txt",
    # "finetune_1000/ema/ema_0.8_10.0.txt",
    # "finetune_1000/ema/ema_0.8_20.0.txt",

    # "finetune_1000/ema/ema_0.9_5.0.txt",
    # "finetune_1000/ema/ema_0.9_10.0.txt",
    # "finetune_1000/ema/ema_0.9_20.0.txt",

    # "finetune_1000/ema/ema_0.5_10.0.txt",
    # "finetune_1000/ema/ema_0.7_10.0.txt",
    # "finetune_1000/ema/ema_0.9_10.0.txt",
    # "finetune_1000/ema/ema_0.95_10.0.txt",


    # **************************************Heavyball Finetune 1000********************************
    # "finetune_1000/heavyball/beta=0.9, ηs=0.5.txt",
    # "finetune_1000/heavyball/beta=0.9, ηs=1.0.txt",
    # "finetune_1000/heavyball/beta=0.9, ηs=2.0.txt",

    # "finetune_1000/heavyball/beta=0.8, ηs=1.0.txt",
    # "finetune_1000/heavyball/beta=0.9, ηs=1.0.txt",
    # "finetune_1000/heavyball/beta=0.99, ηs=1.0.txt",

    # "finetune_1000/heavyball/heavyball_0.99_2.0.txt",

    # **************************************AB test 1000********************************
    # "AB_test_1000/LoRA-FixedA.txt",
    # "AB_test_1000/LoRA-FixedB.txt",

    # "AB_test_1000/LoRA-MomA.txt",
    # "AB_test_1000/LoRA-MomB.txt",

    # "AB_test_1000/LoRA-MomB-FixedA.txt",
    # "AB_test_1000/LoRA-MomA-FixedB.txt",

    # "AB_test_1000/LoRA-FixedB.txt",
    # "AB_test_1000/LoRA-MomA.txt",
    # "AB_test_1000/LoRA-MomA-FixedB.txt",

    # **************************************AB test********************************
    # "AB_test/fedncf_fixedA.txt",
    # "AB_test/fedncf_fixedB.txt",

    # "AB_test/fedncf_momentumA.txt",
    # "AB_test/fedncf_momentumB.txt",

    # "AB_test/fedncf_momentumB_fixedA.txt",
    # "AB_test/fedncf_momentumA_fixedB.txt",

    # "AB_test/fedncf_fixedB.txt",
    # "AB_test/fedncf_momentumA.txt",
    # "AB_test/fedncf_momentumA_fixedB.txt",



    # ********************************Finetune Best Result*****************************
    # "finetune/heavyball/Heavy_ball_0.99_2.0.txt",
    # "finetune/ema/EMA_normalization_0.8_20.0.txt",
    # "finetune/adam/Adam_0.005_0.99_0.99.txt",

    # ********************************Adam*****************************
    # "finetune/adam/Adam_0.5_0.9_0.99.txt",
    # "finetune/adam/Adam_0.1_0.9_0.999.txt",
    # "finetune/adam/Adam_0.05_0.9_0.999.txt",
    # "finetune/adam/Adam_0.1_0.95_0.999.txt",
    # "finetune/adam/Adam_0.05_0.95_0.999.txt", #0.05>0.1 0.95>0.9
    # "finetune/adam/Adam_0.05_0.95_0.99.txt",

    # "finetune/adam/Adam_0.05_0.99_0.9.txt", # 0.05>0.1 0.99>0.95>0.9 0.9==0.99


    # "finetune/adam/Adam_0.05_0.99_0.99.txt",
    # "finetune/adam/Adam_0.01_0.99_0.99.txt",
    # "finetune/adam/Adam_0.005_0.99_0.99.txt", #best

    # ********************************Heavyball*****************************
    # "finetune/heavyball/Heavy_ball_0.9_0.5.txt",
    # "finetune/heavyball/Heavy_ball_0.9_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.9_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.9_3.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_0.5.txt",
    # "finetune/heavyball/Heavy_ball_0.99_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_2.0.txt", #best
    # "finetune/heavyball/Heavy_ball_0.99_3.0.txt",

    # "finetune/heavyball/Heavy_ball_0.99_0.5.txt",
    # "finetune/heavyball/Heavy_ball_0.99_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_3.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_5.0.txt",

    # "finetune/heavyball/Heavy_ball_0.5_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.7_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.9_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.95_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_1.0.txt",

    # "finetune/heavyball/Heavy_ball_0.5_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.7_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.9_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.95_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.999_2.0.txt",


    # ********************************EMA Normalization*****************************
    # "finetune/ema/EMA_normalization_0.8_1.0.txt",
    # "finetune/ema/EMA_normalization_0.8_10.0.txt",
    # "finetune/ema/EMA_normalization_0.8_20.0.txt", #best

    # "finetune/ema/EMA_normalization_0.95_1.0.txt",
    # "finetune/ema/EMA_normalization_0.95_10.0.txt",
    # "finetune/ema/EMA_normalization_0.95_20.0.txt",

    # "finetune/ema/EMA_normalization_0.9_1.0.txt",
    # "finetune/ema/EMA_normalization_0.9_10.0.txt",
    # "finetune/ema/EMA_normalization_0.9_20.0.txt",
    # "finetune/ema/EMA_normalization_0.8_1.0.txt",
    # "finetune/ema/EMA_normalization_0.8_10.0.txt",
    # "finetune/ema/EMA_normalization_0.8_20.0.txt",

    # "finetune/ema/EMA_normalization_0.8_1.0.txt",
    # "finetune/ema/EMA_normalization_0.8_10.0.txt",
    # "finetune/ema/EMA_normalization_0.8_20.0.txt",
    # "finetune/ema/EMA_normalization_0.8_30.0.txt",
    # "finetune/ema/EMA_normalization_0.8_50.0.txt",

    # "finetune/ema/EMA_normalization_0.95_20.0.txt",
    # "finetune/ema/EMA_normalization_0.9_20.0.txt",
    # "finetune/ema/EMA_normalization_0.8_20.0.txt",
    # "finetune/ema/EMA_normalization_0.7_20.0.txt",
    # "finetune/ema/EMA_normalization_0.5_20.0.txt",
    # "finetune/ema/EMA_normalization_0.1_20.0.txt",

    # "finetune/ema/EMA_normalization_0.7_20.0.txt",
    # "finetune/ema/abandon/EMA_normalization_0.7_10.0.txt",
    # "finetune/ema/abandon/EMA_normalization_0.5_1.0.txt",

    # "finetune/ema/abandon/EMA_normalization_0.5_50.0.txt",
    # "finetune/ema/abandon/EMA_normalization_0.1_1.0.txt",
    # "finetune/ema/abandon/EMA_normalization_0.1_50.0.txt",



]

# Select which eval metrics to plot (from logs [Metrics] line)
# Supported examples:
# "logloss", "MRR",
# "NDCG(5)", "HR(5)",
# "NDCG(10)", "HR(10)",
# "NDCG(20)", "HR(20)",
# "NDCG(50)", "HR(50)"
SELECTED_METRICS = [
    # "HR(20)",
    # "NDCG(20)",
    # "MRR",
    # "logloss",
    "HR(10)",
    "NDCG(10)",

]

# All known metric options (for validation/help)
AVAILABLE_METRICS = [
    "logloss", "MRR",
    "NDCG(5)", "HR(5)",
    "NDCG(10)", "HR(10)",
    "NDCG(20)", "HR(20)",
    "NDCG(50)", "HR(50)",
]








# def plot_metric(data, x_key, y_key, title, y_label, out_path):
#     plt.figure(figsize=(5, 5))
#     for name, d in data.items():
#         x = d.get(x_key, np.array([]))
#         y = d.get(y_key, np.array([]))
#         if len(x) == 0 or len(y) == 0:
#             continue
#         m = min(len(x), len(y))
#         plt.plot(x[:m], y[:m], linewidth=2, marker="o", markersize=3, label=name)

#     plt.xlabel("round", fontsize=15)
#     plt.ylabel(y_label, fontsize=15)
#     plt.grid(alpha=0.3)
#     plt.legend(fontsize=15)
#     plt.tight_layout()
#     plt.savefig(out_path, bbox_inches="tight")  # pdf works directly by file suffix
#     plt.close()






def _safe_name(metric: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", metric).strip("_").lower()


def main():
    files = SELECTED_LOG_FILES

    if not (1 <= len(files) <= 10):
        raise ValueError("Please provide 1 to 10 file names in SELECTED_LOG_FILES.")

    if len(files) > 5:
        print(f"[Note] You selected {len(files)} files. Plot may look crowded (recommended <= 5).")

    # validate selected metrics
    unknown = [m for m in SELECTED_METRICS if m not in AVAILABLE_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics in SELECTED_METRICS: {unknown}\nAvailable: {AVAILABLE_METRICS}")

    selected_paths = []
    for fname in files:
        p = LOG_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        selected_paths.append(p)

    data = {p.stem: parse_log(p) for p in selected_paths}

    # Smooth loss per file
    for name, d in data.items():
        ys, off = smooth(d["train_losses"], w=9)
        xs = d["train_turns"][off:off + len(ys)] if len(ys) > 0 else np.array([])
        xs_10, ys_10 = sample_every_n_rounds(xs, ys, n=10)
        d["train_turns_smooth"] = xs_10
        d["train_losses_smooth"] = ys_10

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Loss in one figure
    plot_metric(
        data={k: {"train_turns": v["train_turns_smooth"], "train_losses": v["train_losses_smooth"]} for k, v in data.items()},
        x_key="train_turns",
        y_key="train_losses",
        title="Training Loss",
        y_label="Loss",
        out_path=FIG_DIR / f"{ts}_loss_comparison.pdf",
    )

    # 2) Selected eval metrics (one figure per metric)
    for metric in SELECTED_METRICS:
        metric_plot_data = {}
        for run_name, d in data.items():
            metric_dict = d.get("eval_metrics", {})
            if metric in metric_dict:
                metric_plot_data[run_name] = {
                    "eval_turns": metric_dict[metric]["turns"],
                    "metric_values": metric_dict[metric]["values"],
                }

        if len(metric_plot_data) == 0:
            print(f"[Warning] Metric '{metric}' not found in selected logs. Skip.")
            continue

        safe_metric = _safe_name(metric)
        plot_metric(
            data=metric_plot_data,
            x_key="eval_turns",
            y_key="metric_values",
            title=f"{metric}",
            y_label=metric,
            out_path=FIG_DIR / f"{ts}_{safe_metric}_comparison.pdf",
        )

    print(f"Saved to: {FIG_DIR}")
    print(f"- {ts}_loss_comparison.pdf")
    for metric in SELECTED_METRICS:
        print(f"- {ts}_{_safe_name(metric)}_comparison.pdf")


if __name__ == "__main__":
    main()
