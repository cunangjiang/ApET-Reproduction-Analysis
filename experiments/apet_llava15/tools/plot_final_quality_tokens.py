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

vanilla = df[
    df["Setting"] == "Vanilla-576"
].iloc[0]

apet = (
    df[df["Method"] == "ApET"]
    .sort_values("Avg Visual Tokens")
    .copy()
)

rand = df[
    df["Setting"] == "RandomPrune-64"
].iloc[0]


def build_apet_trajectory(metric):
    rows = []

    # ApET 64 / 128 / 192
    for _, r in apet.iterrows():
        rows.append({
            "setting": r["Setting"],
            "tokens": r["Avg Visual Tokens"],
            "value": r[metric],
        })

    # Vanilla 576
    rows.append({
        "setting": "Vanilla-576",
        "tokens": vanilla["Avg Visual Tokens"],
        "value": vanilla[metric],
    })

    return (
        pd.DataFrame(rows)
        .sort_values("tokens")
        .reset_index(drop=True)
    )


fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.2, 4.8)
)


# =========================================================
# (a) GQA
# =========================================================
ax = axes[0]

traj = build_apet_trajectory(
    "GQA Accuracy (%)"
)

ax.plot(
    traj["tokens"],
    traj["value"],
    marker="o",
    markersize=7,
    linewidth=2.2,
    label="Vanilla → ApET",
    zorder=3,
)

ax.scatter(
    rand["Avg Visual Tokens"],
    rand["GQA Accuracy (%)"],
    marker="X",
    s=150,
    linewidths=2.2,
    label="RandomPrune-64",
    zorder=6,
)

offsets = {
    "ApET-64": (10, 10),
    "ApET-128": (8, 7),
    "ApET-192": (8, 7),
    "Vanilla-576": (-78, 7),
}

for _, r in traj.iterrows():
    ax.annotate(
        r["setting"],
        (r["tokens"], r["value"]),
        xytext=offsets[r["setting"]],
        textcoords="offset points",
        fontsize=9,
        zorder=7,
    )

ax.annotate(
    "RandomPrune-64",
    (
        rand["Avg Visual Tokens"],
        rand["GQA Accuracy (%)"],
    ),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=9,
    zorder=7,
)

ax.set_xlabel(
    "Average Visual Tokens",
    fontsize=11,
)

ax.set_ylabel(
    "GQA Accuracy (%)",
    fontsize=11,
)

ax.set_title(
    "(a) GQA",
    fontsize=13,
)

ax.set_xticks(
    [64, 128, 192, 576]
)

ax.grid(
    alpha=0.25
)

ax.margins(
    x=0.06,
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

traj = build_apet_trajectory(
    "POPE Macro-F1 (%)"
)

ax.plot(
    traj["tokens"],
    traj["value"],
    marker="o",
    markersize=7,
    linewidth=2.2,
    label="Vanilla → ApET",
    zorder=3,
)

ax.scatter(
    rand["Avg Visual Tokens"],
    rand["POPE Macro-F1 (%)"],
    marker="X",
    s=150,
    linewidths=2.2,
    label="RandomPrune-64",
    zorder=6,
)

offsets = {
    "ApET-64": (10, 9),
    "ApET-128": (8, 7),
    "ApET-192": (8, 8),
    "Vanilla-576": (-78, 7),
}

for _, r in traj.iterrows():
    ax.annotate(
        r["setting"],
        (r["tokens"], r["value"]),
        xytext=offsets[r["setting"]],
        textcoords="offset points",
        fontsize=9,
        zorder=7,
    )

ax.annotate(
    "RandomPrune-64",
    (
        rand["Avg Visual Tokens"],
        rand["POPE Macro-F1 (%)"],
    ),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=9,
    zorder=7,
)

ax.set_xlabel(
    "Average Visual Tokens",
    fontsize=11,
)

ax.set_ylabel(
    "POPE Macro-F1 (%)",
    fontsize=11,
)

ax.set_title(
    "(b) POPE",
    fontsize=13,
)

ax.set_xticks(
    [64, 128, 192, 576]
)

ax.grid(
    alpha=0.25
)

ax.margins(
    x=0.06,
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
    "final_quality_vs_tokens.png"
)

pdf_path = (
    FIG_DIR /
    "final_quality_vs_tokens.pdf"
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
