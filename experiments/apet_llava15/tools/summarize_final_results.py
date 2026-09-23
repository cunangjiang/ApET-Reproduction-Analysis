from pathlib import Path
import pandas as pd

EXP = Path("experiments/apet_llava15")
METRICS = EXP / "metrics"

df = pd.read_csv(
    METRICS / "final_controlled_results.csv"
)

def get(setting):
    return df[df["Setting"] == setting].iloc[0]


v = get("Vanilla-576")
a192 = get("ApET-192")
a128 = get("ApET-128")
a64 = get("ApET-64")
r64 = get("RandomPrune-64")


summary = {
    "ApET64_vs_Vanilla_token_reduction_pct":
        (1 - 64 / 576) * 100,

    "ApET64_vs_Vanilla_GQA_change_pp":
        a64["GQA Accuracy (%)"]
        - v["GQA Accuracy (%)"],

    "ApET64_vs_Vanilla_POPE_change_pp":
        a64["POPE Macro-F1 (%)"]
        - v["POPE Macro-F1 (%)"],

    "ApET64_vs_Vanilla_E2E_reduction_pct":
        (
            1
            - a64["E2E Latency Mean (ms)"]
            / v["E2E Latency Mean (ms)"]
        ) * 100,

    "ApET64_vs_Vanilla_E2E_speedup":
        a64["E2E Speedup"],

    "ApET64_vs_Vanilla_prefill_reduction_pct":
        (
            1
            - a64["LLM Prefill Mean (ms)"]
            / v["LLM Prefill Mean (ms)"]
        ) * 100,

    "ApET64_vs_Random_GQA_gain_pp":
        a64["GQA Accuracy (%)"]
        - r64["GQA Accuracy (%)"],

    "ApET64_vs_Random_POPE_gain_pp":
        a64["POPE Macro-F1 (%)"]
        - r64["POPE Macro-F1 (%)"],

    "ApET64_vs_Random_E2E_overhead_ms":
        a64["E2E Latency Mean (ms)"]
        - r64["E2E Latency Mean (ms)"],

    "ApET64_vs_Random_E2E_overhead_pct":
        (
            a64["E2E Latency Mean (ms)"]
            - r64["E2E Latency Mean (ms)"]
        )
        / r64["E2E Latency Mean (ms)"]
        * 100,

    "ApET64_vs_Random_prefill_overhead_ms":
        a64["LLM Prefill Mean (ms)"]
        - r64["LLM Prefill Mean (ms)"],

    "ApET64_vs_Random_prefill_overhead_pct":
        (
            a64["LLM Prefill Mean (ms)"]
            - r64["LLM Prefill Mean (ms)"]
        )
        / r64["LLM Prefill Mean (ms)"]
        * 100,
}

out = pd.DataFrame(
    summary.items(),
    columns=["Metric", "Value"]
)

path = METRICS / "final_key_findings.csv"
out.to_csv(path, index=False)

print("\n========== KEY FINDINGS ==========\n")

for k, v in summary.items():
    print(f"{k:<48s}: {v:.3f}")

print("\nSaved:", path)
