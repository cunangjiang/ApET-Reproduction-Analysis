from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


EXP = Path("experiments/apet_llava15")
METRICS = EXP / "metrics"
FIG_DIR = EXP / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    METRICS / "final_controlled_results.csv"
)


# Compression trajectory:
# Vanilla -> ApET192 -> ApET128 -> ApET64
order = [
    "Vanilla-576",
    "ApET-192",
    "ApET-128",
    "ApET-64",
]

traj = df[
    df["Setting"].isin(order)
].copy()

order_map = {
    name: i
    for i, name in enumerate(order)
}

traj["_order"] = (
    traj["Setting"].map(order_map)
)

traj = (
    traj
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

rand = df[
    df["Setting"] == "RandomPrune-64"
].iloc[0]


fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.2, 4.8)
)


# =========================================================
# (a) GQA
# =========================================================
ax = axes[0]

ax.errorbar(
    traj["E2E Latency Mean (ms)"],
    traj["GQA Accuracy (%)"],
    xerr=traj["E2E Latency Std (ms)"],
    marker="o",
    markersize=7,
    linewidth=2.2,
    capsize=3,
    label="Vanilla → ApET",
    zorder=3,
)

ax.errorbar(
    rand["E2E Latency Mean (ms)"],
    rand["GQA Accuracy (%)"],
    xerr=rand["E2E Latency Std (ms)"],
    marker="X",
    markersize=10,
    markeredgewidth=2.0,
    linestyle="none",
    capsize=3,
    label="RandomPrune-64",
    zorder=6,
)

offsets = {
    "Vanilla-576": (-76, 7),
    "ApET-192": (8, 7),
    "ApET-128": (8, 7),
    "ApET-64": (8, 8),
}

for _, r in traj.iterrows():
    ax.annotate(
        r["Setting"],
        (
            r["E2E Latency Mean (ms)"],
            r["GQA Accuracy (%)"],
        ),
        xytext=offsets[r["Setting"]],
        textcoords="offset points",
        fontsize=9,
        zorder=7,
    )

ax.annotate(
    "RandomPrune-64",
    (
        rand["E2E Latency Mean (ms)"],
        rand["GQA Accuracy (%)"],
    ),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=9,
    zorder=7,
)

ax.set_xlabel(
    "E2E Model Latency (ms / sample)",
    fontsize=11,
)

ax.set_ylabel(
    "GQA Accuracy (%)",
    fontsize=11,
)

ax.set_title(
    "(a) GQA quality–latency trade-off",
    fontsize=13,
)

ax.grid(
    alpha=0.25
)

ax.margins(
    x=0.07,
    y=0.10,
)

ax.legend(
    fontsize=9,
    loc="upper left",
    frameon=True,
)


# =========================================================
# (b) POPE
# =========================================================
ax = axes[1]

ax.errorbar(
    traj["E2E Latency Mean (ms)"],
    traj["POPE Macro-F1 (%)"],
    xerr=traj["E2E Latency Std (ms)"],
    marker="o",
    markersize=7,
    linewidth=2.2,
    capsize=3,
    label="Vanilla → ApET",
    zorder=3,
)

ax.errorbar(
    rand["E2E Latency Mean (ms)"],
    rand["POPE Macro-F1 (%)"],
    xerr=rand["E2E Latency Std (ms)"],
    marker="X",
    markersize=10,
    markeredgewidth=2.0,
    linestyle="none",
    capsize=3,
    label="RandomPrune-64",
    zorder=6,
)

offsets = {
    "Vanilla-576": (-76, 7),
    "ApET-192": (8, 7),
    "ApET-128": (8, 7),
    "ApET-64": (8, 8),
}

for _, r in traj.iterrows():
    ax.annotate(
        r["Setting"],
        (
            r["E2E Latency Mean (ms)"],
            r["POPE Macro-F1 (%)"],
        ),
        xytext=offsets[r["Setting"]],
        textcoords="offset points",
        fontsize=9,
        zorder=7,
    )

ax.annotate(
    "RandomPrune-64",
    (
        rand["E2E Latency Mean (ms)"],
        rand["POPE Macro-F1 (%)"],
    ),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=9,
    zorder=7,
)

ax.set_xlabel(
    "E2E Model Latency (ms / sample)",
    fontsize=11,
)

ax.set_ylabel(
    "POPE Macro-F1 (%)",
    fontsize=11,
)

ax.set_title(
    "(b) POPE quality–latency trade-off",
    fontsize=13,
)

ax.grid(
    alpha=0.25
)

ax.margins(
    x=0.07,
    y=0.10,
)

ax.legend(
    fontsize=9,
    loc="lower right",
    frameon=True,
)


plt.tight_layout(
    w_pad=2.2
)

png_path = (
    FIG_DIR /
    "final_quality_latency_tradeoff.png"
)

pdf_path = (
    FIG_DIR /
    "final_quality_latency_tradeoff.pdf"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.close()

print("Saved:")
print(png_path)
print(pdf_path)
