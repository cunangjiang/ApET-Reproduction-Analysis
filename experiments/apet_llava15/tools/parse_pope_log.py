import csv
import re
import sys
from pathlib import Path

run_id = sys.argv[1]
log_path = Path(sys.argv[2])
csv_path = Path(sys.argv[3])

text = log_path.read_text()

rows = []

blocks = re.split(r"Category:\s*", text)[1:]

for block in blocks:
    first = block.splitlines()[0]
    category = first.split(",")[0].strip()

    m_n = re.search(r"# samples:\s*(\d+)", first)
    m_acc = re.search(r"Accuracy:\s*([0-9.]+)", block)
    m_pre = re.search(r"Precision:\s*([0-9.]+)", block)
    m_rec = re.search(r"Recall:\s*([0-9.]+)", block)
    m_f1 = re.search(r"F1 score:\s*([0-9.]+)", block)
    m_yes = re.search(r"Yes ratio:\s*([0-9.]+)", block)

    if not all([m_n, m_acc, m_pre, m_rec, m_f1, m_yes]):
        continue

    rows.append({
        "run_id": run_id,
        "category": category,
        "samples": int(m_n.group(1)),
        "accuracy": float(m_acc.group(1)),
        "precision": float(m_pre.group(1)),
        "recall": float(m_rec.group(1)),
        "f1": float(m_f1.group(1)),
        "yes_ratio": float(m_yes.group(1)),
    })

if not rows:
    raise RuntimeError("No POPE metrics parsed from log.")

# 添加三类的 macro average
rows.append({
    "run_id": run_id,
    "category": "macro_avg",
    "samples": sum(x["samples"] for x in rows),
    "accuracy": sum(x["accuracy"] for x in rows) / len(rows),
    "precision": sum(x["precision"] for x in rows) / len(rows),
    "recall": sum(x["recall"] for x in rows) / len(rows),
    "f1": sum(x["f1"] for x in rows) / len(rows),
    "yes_ratio": sum(x["yes_ratio"] for x in rows) / len(rows),
})

fields = [
    "run_id", "category", "samples",
    "accuracy", "precision", "recall", "f1", "yes_ratio"
]

existing = []
if csv_path.exists():
    with csv_path.open(newline="") as f:
        existing = list(csv.DictReader(f))

# 如果同一个 run 重跑，先删除旧记录，防止重复
existing = [x for x in existing if x["run_id"] != run_id]

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(existing)
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {csv_path}")
