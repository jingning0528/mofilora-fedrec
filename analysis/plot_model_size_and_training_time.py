import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ==========================================================
# Data
# ==========================================================
methods = ["Full", "LoRA", "LoRA-FixedB", "MoFiLoRA"]

model_size = [0.4824, 0.0679, 0.0603, 0.0603]

local_time = [1487.08, 1824.72, 1716.77, 1695.60]
aggregation_time = [56.99, 67.04, 64.27, 64.37]

# ==========================================================
# Output path
# ==========================================================
save_path = str(Path(__file__).resolve().parents[1] / "figures" / "Efficiency")
os.makedirs(save_path, exist_ok=True)

# ==========================================================
# IEEE-style formatting
# ==========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

x = np.arange(len(methods))

# ==========================================================
# Figure 1: Model size
# ==========================================================
fig, ax = plt.subplots(figsize=(3.35, 2.35))

colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

bars = ax.bar(
    x,
    model_size,
    width=0.60,
    color=colors,
    edgecolor="black",
    linewidth=0.6
)

ax.set_ylabel("Model Size (MB)")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha="right")

ax.set_ylim(0, 0.56)
ax.yaxis.set_major_locator(MultipleLocator(0.1))

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)
ax.set_axisbelow(True)

for bar, value in zip(bars, model_size):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.012,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=6
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    os.path.join(save_path, "model_size.pdf"),
    bbox_inches="tight"
)

plt.close(fig)

# ==========================================================
# Figure 2: Computation time
# ==========================================================
fig, ax = plt.subplots(figsize=(3.35, 2.35))

width = 0.34

bars_local = ax.bar(
    x - width / 2,
    local_time,
    width=width,
    label="Local",
    color="#4C78A8",
    edgecolor="black",
    linewidth=0.6
)

bars_aggregation = ax.bar(
    x + width / 2,
    aggregation_time,
    width=width,
    label="Aggregation",
    color="#F58518",
    edgecolor="black",
    linewidth=0.6,
    hatch="//"
)

ax.set_ylabel("Time (s)")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha="right")

ax.set_ylim(0, 2100)
ax.yaxis.set_major_locator(MultipleLocator(400))

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)
ax.set_axisbelow(True)

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    handlelength=1.4,
    handletextpad=0.4,
    columnspacing=1.0
)

for bar, value in zip(bars_local, local_time):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 30,
        f"{value:.0f}",
        ha="center",
        va="bottom",
        fontsize=6
    )

for bar, value in zip(bars_aggregation, aggregation_time):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 18,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=6
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    os.path.join(save_path, "computation_time.pdf"),
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:")
print(os.path.join(save_path, "model_size.pdf"))
print(os.path.join(save_path, "computation_time.pdf"))