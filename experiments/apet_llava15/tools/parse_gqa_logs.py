import re
import csv
from pathlib import Path

exp = Path("experiments/apet_llava15")

runs = [
    ("gqa_vanilla_576_s42", "Vanilla", 576, 576, 576),
    ("gqa_apet_192_s42",    "ApET",    192, 288,  96),
    ("gqa_apet_128_s42",    "ApET",    128, 192,  64),
    ("gqa_apet_64_s42",     "ApET",     64,  96,  32),
]

rows = []

def get(pattern, text):
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None

for run_id, method, avg_tokens, vision_tokens, layer16_tokens in runs:
    log = exp / "gqa/eval" / f"{run_id}.txt"

    if not log.exists():
        raise FileNotFoundError(log)

    text = log.read_text(errors="ignore")

    binary = get(r"^Binary:\s*([0-9.]+)%", text)
    open_acc = get(r"^Open:\s*([0-9.]+)%", text)
    accuracy = get(r"^Accuracy:\s*([0-9.]+)%", text)
    validity = get(r"^Validity:\s*([0-9.]+)%", text)
    plausibility = get(r"^Plausibility:\s*([0-9.]+)%", text)
    distribution = get(r"^Distribution:\s*([0-9.]+)", text)

    if accuracy is None:
        raise RuntimeError(f"Could not parse Accuracy from {log}")

    rows.append({
        "run_id": run_id,
        "method": method,
        "avg_visual_tokens": avg_tokens,
        "vision_stage_tokens": vision_tokens,
        "layer16_tokens": layer16_tokens,
        "binary_pct": binary,
        "open_pct": open_acc,
        "accuracy_pct": accuracy,
        "validity_pct": validity,
        "plausibility_pct": plausibility,
        "distribution": distribution,
    })

out = exp / "metrics/gqa_scores.csv"

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nGQA RESULTS")
print("=" * 90)

for r in rows:
    print(
        f'{r["run_id"]:26s} '
        f'AvgTokens={r["avg_visual_tokens"]:3d}  '
        f'Accuracy={r["accuracy_pct"]:6.2f}%  '
        f'Binary={r["binary_pct"]!s:>6}  '
        f'Open={r["open_pct"]!s:>6}'
    )

print("\nSaved:", out)
