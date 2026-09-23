from pathlib import Path
import pandas as pd

EXP = Path("experiments/apet_llava15")
METRICS = EXP / "metrics"

main = pd.read_csv(
    METRICS / "final_main_results.csv"
)

rand_q = pd.read_csv(
    METRICS / "random_prune_64_results.csv"
)

rand_e = pd.read_csv(
    METRICS / "randomprune_efficiency_aggregate.csv"
)

rows = []

# ---------------------------------------------------------
# Vanilla + ApET-192/128/64
# ---------------------------------------------------------
for _, r in main.sort_values(
    "avg_visual_tokens",
    ascending=False
).iterrows():

    setting = (
        "Vanilla-576"
        if int(r["avg_visual_tokens"]) == 576
        else f"ApET-{int(r['avg_visual_tokens'])}"
    )

    rows.append({
        "Setting": setting,
        "Method": r["method"],
        "Avg Visual Tokens": int(r["avg_visual_tokens"]),
        "Retention (%)": r["avg_visual_tokens"] / 576 * 100,

        "GQA Accuracy (%)":
            r["gqa_accuracy_pct"],

        "POPE Macro-F1 (%)":
            r["pope_f1_pct"],

        "E2E Latency Mean (ms)":
            r["e2e_mean_ms"],

        "E2E Latency Std (ms)":
            r["e2e_across_run_std_ms"],

        "E2E Speedup":
            r["e2e_speedup_mean"],

        "LLM Prefill Mean (ms)":
            r["prefill_mean_ms"],

        "LLM Prefill Std (ms)":
            r["prefill_across_run_std_ms"],

        "Throughput (samples/s)":
            r["throughput_mean_samples_per_s"],

        "Source":
            "Our controlled experiment",
    })


# ---------------------------------------------------------
# RandomPrune-64
# ---------------------------------------------------------
rq = rand_q.iloc[0]
re = rand_e.iloc[0]

rows.append({
    "Setting": "RandomPrune-64",
    "Method": "RandomPrune",
    "Avg Visual Tokens": 64,
    "Retention (%)": 64 / 576 * 100,

    "GQA Accuracy (%)":
        rq["gqa_accuracy_pct"],

    "POPE Macro-F1 (%)":
        rq["pope_f1_pct"],

    "E2E Latency Mean (ms)":
        re["e2e_mean_ms"],

    "E2E Latency Std (ms)":
        re["e2e_std_ms"],

    "E2E Speedup":
        re["e2e_speedup_mean"],

    "LLM Prefill Mean (ms)":
        re["prefill_mean_ms"],

    "LLM Prefill Std (ms)":
        re["prefill_std_ms"],

    "Throughput (samples/s)":
        re["throughput_mean"],

    "Source":
        "Our controlled experiment",
})


out = pd.DataFrame(rows)

order = {
    "Vanilla-576": 0,
    "ApET-192": 1,
    "ApET-128": 2,
    "RandomPrune-64": 3,
    "ApET-64": 4,
}

out["_order"] = out["Setting"].map(order)
out = out.sort_values("_order").drop(columns="_order")

path = METRICS / "final_controlled_results.csv"
out.to_csv(path, index=False)

print("\n========== FINAL CONTROLLED RESULTS ==========\n")

show = out.copy()

for c in [
    "Retention (%)",
    "GQA Accuracy (%)",
    "POPE Macro-F1 (%)",
    "E2E Latency Mean (ms)",
    "E2E Latency Std (ms)",
    "LLM Prefill Mean (ms)",
    "LLM Prefill Std (ms)",
    "Throughput (samples/s)",
]:
    show[c] = show[c].map(lambda x: round(float(x), 2))

show["E2E Speedup"] = (
    show["E2E Speedup"]
    .map(lambda x: round(float(x), 3))
)

print(show.to_string(index=False))

print("\nSaved:")
print(path)
