import json
from pathlib import Path

import pandas as pd

EXP = Path("experiments/apet_llava15")

RAND_ROOT = (
    EXP /
    "random_prune/efficiency/raw"
)

MAIN_SCORES = (
    EXP /
    "metrics/efficiency_scores.csv"
)

OUT = (
    EXP /
    "metrics/randomprune_efficiency.csv"
)

main = pd.read_csv(MAIN_SCORES)

vanilla = (
    main[
        main["setting"] == "Vanilla-576"
    ]
    .set_index("repeat")
)

rows = []

for rep in ["r1", "r2", "r3"]:

    p = (
        RAND_ROOT /
        f"eff_randomprune_64_{rep}_summary.json"
    )

    x = json.load(open(p))

    assert x["measured_samples"] == 500

    vanilla_e2e = vanilla.loc[
        rep,
        "e2e_mean_ms"
    ]

    vanilla_prefill = vanilla.loc[
        rep,
        "prefill_mean_ms"
    ]

    rows.append({
        "setting": "RandomPrune-64",
        "method": "RandomPrune",
        "repeat": rep,

        "avg_visual_tokens": 64,
        "vision_stage_tokens": 96,
        "layer16_tokens": 32,

        "e2e_mean_ms":
            x["generation_mean_ms"],

        "prefill_mean_ms":
            x["prefill_mean_ms"],

        "throughput_samples_per_s":
            x["throughput_samples_per_s"],

        "peak_allocated_gb":
            x["peak_allocated_gb"],

        "e2e_speedup_vs_vanilla":
            vanilla_e2e /
            x["generation_mean_ms"],

        "prefill_speedup_vs_vanilla":
            vanilla_prefill /
            x["prefill_mean_ms"],
    })


df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)


summary = {
    "setting": "RandomPrune-64",
    "avg_visual_tokens": 64,

    "e2e_mean_ms":
        df["e2e_mean_ms"].mean(),

    "e2e_std_ms":
        df["e2e_mean_ms"].std(ddof=1),

    "e2e_speedup_mean":
        df["e2e_speedup_vs_vanilla"].mean(),

    "e2e_speedup_std":
        df["e2e_speedup_vs_vanilla"].std(ddof=1),

    "prefill_mean_ms":
        df["prefill_mean_ms"].mean(),

    "prefill_std_ms":
        df["prefill_mean_ms"].std(ddof=1),

    "prefill_speedup_mean":
        df["prefill_speedup_vs_vanilla"].mean(),

    "prefill_speedup_std":
        df["prefill_speedup_vs_vanilla"].std(ddof=1),

    "throughput_mean":
        df["throughput_samples_per_s"].mean(),

    "throughput_std":
        df["throughput_samples_per_s"].std(ddof=1),

    "peak_allocated_gb":
        df["peak_allocated_gb"].mean(),
}

summary_df = pd.DataFrame([summary])

summary_path = (
    EXP /
    "metrics/randomprune_efficiency_aggregate.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\n========== RAW RANDOMPRUNE RUNS ==========\n")

print(
    df[
        [
            "repeat",
            "e2e_mean_ms",
            "prefill_mean_ms",
            "throughput_samples_per_s",
            "e2e_speedup_vs_vanilla",
        ]
    ].to_string(index=False)
)

print("\n========== RANDOMPRUNE AGGREGATE ==========\n")

r = summary

print(
    f"RandomPrune-64 "
    f"E2E {r['e2e_mean_ms']:.2f} "
    f"± {r['e2e_std_ms']:.2f} ms"
)

print(
    f"Speedup "
    f"{r['e2e_speedup_mean']:.3f} "
    f"± {r['e2e_speedup_std']:.3f}x"
)

print(
    f"LLM Prefill "
    f"{r['prefill_mean_ms']:.2f} "
    f"± {r['prefill_std_ms']:.2f} ms"
)

print(
    f"Prefill speedup "
    f"{r['prefill_speedup_mean']:.3f} "
    f"± {r['prefill_speedup_std']:.3f}x"
)

print(
    f"Throughput "
    f"{r['throughput_mean']:.3f} "
    f"± {r['throughput_std']:.3f} samples/s"
)

print("\nSaved:")
print(OUT)
print(summary_path)
