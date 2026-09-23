import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

exp = Path("experiments/apet_llava15")
figdir = exp / "figures"
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    exp / "metrics/final_main_results.csv"
)

df = df.sort_values("avg_visual_tokens")

x = df["avg_visual_tokens"]

# ==========================================================
# Figure 1: Quality vs Visual Tokens
# ==========================================================
plt.figure(figsize=(6.4, 4.6))

plt.plot(
    x,
    df["gqa_accuracy_pct"],
    marker="o",
    linewidth=2,
    label="GQA Accuracy",
)

plt.plot(
    x,
    df["pope_f1_pct"],
    marker="s",
    linewidth=2,
    label="POPE Macro-F1",
)

plt.xlabel("Average Visual Tokens")
plt.ylabel("Score (%)")
plt.xticks([64, 128, 192, 576])
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    figdir / "quality_vs_tokens.pdf",
    bbox_inches="tight",
)

plt.savefig(
    figdir / "quality_vs_tokens.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ==========================================================
# Figure 2: Latency vs Visual Tokens
# ==========================================================
plt.figure(figsize=(6.4, 4.6))

plt.errorbar(
    x,
    df["e2e_mean_ms"],
    yerr=df["e2e_across_run_std_ms"],
    marker="o",
    linewidth=2,
    capsize=4,
    label="E2E Model Latency",
)

plt.errorbar(
    x,
    df["prefill_mean_ms"],
    yerr=df["prefill_across_run_std_ms"],
    marker="s",
    linewidth=2,
    capsize=4,
    label="LLM Prefill Latency",
)

plt.xlabel("Average Visual Tokens")
plt.ylabel("Latency (ms / sample)")
plt.xticks([64, 128, 192, 576])
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    figdir / "latency_vs_tokens.pdf",
    bbox_inches="tight",
)

plt.savefig(
    figdir / "latency_vs_tokens.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ==========================================================
# Figure 3: GQA quality-efficiency trade-off
# ==========================================================
plt.figure(figsize=(6.4, 4.6))

plt.plot(
    df["e2e_mean_ms"],
    df["gqa_accuracy_pct"],
    marker="o",
    linewidth=2,
)

for _, r in df.iterrows():
    label = (
        "Vanilla-576"
        if r["avg_visual_tokens"] == 576
        else f'ApET-{int(r["avg_visual_tokens"])}'
    )

    plt.annotate(
        label,
        (
            r["e2e_mean_ms"],
            r["gqa_accuracy_pct"],
        ),
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=9,
    )

plt.xlabel("E2E Model Latency (ms / sample)")
plt.ylabel("GQA Accuracy (%)")
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    figdir / "gqa_quality_latency_tradeoff.pdf",
    bbox_inches="tight",
)

plt.savefig(
    figdir / "gqa_quality_latency_tradeoff.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Saved figures to:", figdir)
