import json
from pathlib import Path

import pandas as pd
import numpy as np

EXP = Path("experiments/apet_llava15")
ROOT = EXP / "efficiency" / "raw"
METRICS = EXP / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

configs = [
    {
        "setting": "Vanilla-576",
        "method": "Vanilla",
        "avg_visual_tokens": 576,
        "vision_stage_tokens": 576,
        "layer16_tokens": 576,
    },
    {
        "setting": "ApET-192",
        "method": "ApET",
        "avg_visual_tokens": 192,
        "vision_stage_tokens": 288,
        "layer16_tokens": 96,
    },
    {
        "setting": "ApET-128",
        "method": "ApET",
        "avg_visual_tokens": 128,
        "vision_stage_tokens": 192,
        "layer16_tokens": 64,
    },
    {
        "setting": "ApET-64",
        "method": "ApET",
        "avg_visual_tokens": 64,
        "vision_stage_tokens": 96,
        "layer16_tokens": 32,
    },
]

prefix = {
    "Vanilla-576": "eff_vanilla_576",
    "ApET-192": "eff_apet_192",
    "ApET-128": "eff_apet_128",
    "ApET-64": "eff_apet_64",
}

rows = []

for cfg in configs:
    for rep in ["r1", "r2", "r3"]:
        fn = ROOT / f"{prefix[cfg['setting']]}_{rep}_summary.json"

        if not fn.exists():
            raise FileNotFoundError(f"Missing: {fn}")

        with fn.open() as f:
            x = json.load(f)

        if x["measured_samples"] != 500:
            raise RuntimeError(
                f"{fn}: measured_samples={x['measured_samples']}, expected 500"
            )

        rows.append({
            "setting": cfg["setting"],
            "method": cfg["method"],
            "avg_visual_tokens": cfg["avg_visual_tokens"],
            "vision_stage_tokens": cfg["vision_stage_tokens"],
            "layer16_tokens": cfg["layer16_tokens"],
            "repeat": rep,

            "e2e_mean_ms": x["generation_mean_ms"],
            "e2e_sample_std_ms": x["generation_std_ms"],

            "prefill_mean_ms": x["prefill_mean_ms"],
            "prefill_sample_std_ms": x["prefill_std_ms"],

            "throughput_samples_per_s":
                x["throughput_samples_per_s"],

            "peak_allocated_gb":
                x["peak_allocated_gb"],

            "dynamic_peak_allocated_gb":
                x["dynamic_peak_allocated_gb"],

            "total_generation_seconds":
                x["total_generation_seconds"],

            "measured_samples":
                x["measured_samples"],
        })

df = pd.DataFrame(rows)

# ----------------------------------------------------------
# 每个 repeat 与同一 repeat 的 Vanilla 配对计算 speedup
# ----------------------------------------------------------
vanilla_e2e = (
    df[df["setting"] == "Vanilla-576"]
    .set_index("repeat")["e2e_mean_ms"]
)

vanilla_prefill = (
    df[df["setting"] == "Vanilla-576"]
    .set_index("repeat")["prefill_mean_ms"]
)

df["e2e_speedup"] = df.apply(
    lambda r:
        vanilla_e2e.loc[r["repeat"]] / r["e2e_mean_ms"],
    axis=1,
)

df["prefill_speedup"] = df.apply(
    lambda r:
        vanilla_prefill.loc[r["repeat"]] / r["prefill_mean_ms"],
    axis=1,
)

# 保存12个原始run
scores_path = METRICS / "efficiency_scores.csv"
df.to_csv(scores_path, index=False)

# ----------------------------------------------------------
# 对 r1/r2/r3 的 run-level mean 求 mean ± std
#
# 注意：
# 这里 std 是“3次独立repeat之间的std”，
# 不是每个500-sample run内部的sample std。
# ----------------------------------------------------------
agg_rows = []

for cfg in configs:
    g = df[df["setting"] == cfg["setting"]].copy()

    def mean(col):
        return g[col].mean()

    def std(col):
        return g[col].std(ddof=1)

    agg_rows.append({
        "setting": cfg["setting"],
        "method": cfg["method"],
        "avg_visual_tokens": cfg["avg_visual_tokens"],
        "token_retention_pct":
            cfg["avg_visual_tokens"] / 576 * 100,
        "vision_stage_tokens":
            cfg["vision_stage_tokens"],
        "layer16_tokens":
            cfg["layer16_tokens"],

        "repeats": len(g),
        "samples_per_repeat": 500,

        "e2e_mean_ms": mean("e2e_mean_ms"),
        "e2e_across_run_std_ms": std("e2e_mean_ms"),

        "e2e_speedup_mean": mean("e2e_speedup"),
        "e2e_speedup_std": std("e2e_speedup"),

        "prefill_mean_ms": mean("prefill_mean_ms"),
        "prefill_across_run_std_ms":
            std("prefill_mean_ms"),

        "prefill_speedup_mean":
            mean("prefill_speedup"),
        "prefill_speedup_std":
            std("prefill_speedup"),

        "throughput_mean_samples_per_s":
            mean("throughput_samples_per_s"),
        "throughput_across_run_std":
            std("throughput_samples_per_s"),

        "peak_allocated_gb_mean":
            mean("peak_allocated_gb"),
        "peak_allocated_gb_std":
            std("peak_allocated_gb"),
    })

agg = pd.DataFrame(agg_rows)

agg_path = METRICS / "efficiency_aggregate.csv"
agg.to_csv(agg_path, index=False)

print("\n================ RAW 12 RUNS ================\n")
print(
    df[
        [
            "setting",
            "repeat",
            "e2e_mean_ms",
            "prefill_mean_ms",
            "throughput_samples_per_s",
            "e2e_speedup",
        ]
    ].to_string(index=False)
)

print("\n================ FINAL AGGREGATE ================\n")

for _, r in agg.iterrows():
    print(
        f"{r['setting']:<14}"
        f"  E2E "
        f"{r['e2e_mean_ms']:.2f}"
        f" ± {r['e2e_across_run_std_ms']:.2f} ms"
        f"  | Speedup "
        f"{r['e2e_speedup_mean']:.3f}"
        f" ± {r['e2e_speedup_std']:.3f}x"
        f"  | Prefill "
        f"{r['prefill_mean_ms']:.2f}"
        f" ± {r['prefill_across_run_std_ms']:.2f} ms"
        f"  | Throughput "
        f"{r['throughput_mean_samples_per_s']:.3f}/s"
    )

print("\nSaved:")
print(scores_path)
print(agg_path)
