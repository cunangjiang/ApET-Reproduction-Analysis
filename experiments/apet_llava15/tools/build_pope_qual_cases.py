import json
import shutil
from pathlib import Path

import pandas as pd


EXP = Path("experiments/apet_llava15")

QUESTION_FILE = Path(
    "data/eval/pope/llava_pope_test.jsonl"
)

ANNOTATION_DIR = Path(
    "data/eval/pope/coco"
)

IMAGE_DIR = Path(
    "data/eval/pope/val2014"
)

VANILLA_FILE = (
    EXP /
    "pope/answers/pope_vanilla_576_s42.jsonl"
)

APET_FILE = (
    EXP /
    "pope/answers/pope_apet_64_s42.jsonl"
)

RANDOM_FILE = (
    EXP /
    "random_prune/pope/answers/"
    "pope_randomprune_64_s42.jsonl"
)

OUT = EXP / "cases/pope"
IMG_OUT = OUT / "images"

OUT.mkdir(parents=True, exist_ok=True)
IMG_OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    with open(path) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def normalize_pope(text):
    text = str(text)

    if "." in text:
        text = text.split(".")[0]

    text = text.replace(",", "")
    words = text.split()

    if (
        "No" in words
        or "no" in words
        or "not" in words
    ):
        return "no"

    return "yes"


questions = load_jsonl(QUESTION_FILE)

vanilla_list = load_jsonl(VANILLA_FILE)
apet_list = load_jsonl(APET_FILE)
random_list = load_jsonl(RANDOM_FILE)

assert len(questions) == 8910
assert len(vanilla_list) == 8910
assert len(apet_list) == 8910
assert len(random_list) == 8910


vanilla = {
    x["question_id"]: x
    for x in vanilla_list
}

apet = {
    x["question_id"]: x
    for x in apet_list
}

rand = {
    x["question_id"]: x
    for x in random_list
}


# Recover GT following POPE category order
gt = {}

for category in [
    "popular",
    "random",
    "adversarial",
]:
    qcat = [
        q for q in questions
        if q["category"] == category
    ]

    labels = load_jsonl(
        ANNOTATION_DIR /
        f"coco_pope_{category}.json"
    )

    assert len(qcat) == len(labels)

    for q, ann in zip(qcat, labels):
        gt[q["question_id"]] = (
            ann["label"].lower()
        )


rows = []

for q in questions:

    qid = q["question_id"]

    label = gt[qid]

    v_raw = vanilla[qid]["text"]
    r_raw = rand[qid]["text"]
    a_raw = apet[qid]["text"]

    vp = normalize_pope(v_raw)
    rp = normalize_pope(r_raw)
    ap = normalize_pope(a_raw)

    vc = vp == label
    rc = rp == label
    ac = ap == label

    if vc and not rc and ac:
        pattern = "A_Vcorrect_Rwrong_Acorrect"

    elif vc and not rc and not ac:
        pattern = "B_Vcorrect_Rwrong_Awrong"

    elif not vc and not rc and ac:
        pattern = "C_Vwrong_Rwrong_Acorrect"

    elif vc and rc and not ac:
        pattern = "D_Vcorrect_Rcorrect_Awrong"

    elif vc and rc and ac:
        pattern = "E_all_correct"

    elif not vc and not rc and not ac:
        pattern = "F_all_wrong"

    elif not vc and rc and ac:
        pattern = "G_Vwrong_Rcorrect_Acorrect"

    elif not vc and rc and not ac:
        pattern = "H_only_random_correct"

    else:
        pattern = "OTHER"

    rows.append({
        "question_id": qid,
        "category": q["category"],
        "image": q.get("image", ""),
        "question": q.get("text", ""),
        "gt": label,

        "vanilla_raw": v_raw,
        "random_raw": r_raw,
        "apet_raw": a_raw,

        "vanilla_pred": vp,
        "random_pred": rp,
        "apet_pred": ap,

        "vanilla_correct": vc,
        "random_correct": rc,
        "apet_correct": ac,

        "pattern": pattern,
    })


df = pd.DataFrame(rows)

df.to_csv(
    OUT / "pope_case_pool.csv",
    index=False
)


counts = (
    df.groupby(
        ["pattern", "category"]
    )
    .size()
    .unstack(fill_value=0)
)

counts["total"] = counts.sum(axis=1)

counts.to_csv(
    OUT / "pope_case_counts.csv"
)


# Candidate selection
targets = {
    "A_Vcorrect_Rwrong_Acorrect": 3,
    "B_Vcorrect_Rwrong_Awrong": 1,
    "C_Vwrong_Rwrong_Acorrect": 2,
    "D_Vcorrect_Rcorrect_Awrong": 2,
}

selected = []

for pattern, n in targets.items():

    cand = df[
        df["pattern"] == pattern
    ].copy()

    if len(cand) == 0:
        continue

    take = min(
        n,
        len(cand)
    )

    cand = cand.sample(
        n=take,
        random_state=42
    )

    selected.append(cand)


shortlist = pd.concat(
    selected,
    ignore_index=True
)

letter = {
    "A_Vcorrect_Rwrong_Acorrect": "A",
    "B_Vcorrect_Rwrong_Awrong": "B",
    "C_Vwrong_Rwrong_Acorrect": "C",
    "D_Vcorrect_Rcorrect_Awrong": "D",
}

shortlist.insert(
    0,
    "case_type",
    shortlist["pattern"].map(letter)
)

shortlist.to_csv(
    OUT / "pope_shortlist.csv",
    index=False
)


for _, r in shortlist.iterrows():

    src = (
        IMAGE_DIR /
        str(r["image"])
    )

    if not src.exists():
        print(
            "WARNING missing:",
            src
        )
        continue

    dst = (
        IMG_OUT /
        (
            f"{r['case_type']}_"
            f"{r['category']}_"
            f"{r['question_id']}_"
            f"{src.name}"
        )
    )

    shutil.copy2(
        src,
        dst
    )


print("\n========== CASE COUNTS ==========\n")
print(counts.to_string())

print("\n========== SHORTLIST ==========\n")

print(
    shortlist[
        [
            "case_type",
            "question_id",
            "category",
            "image",
            "question",
            "gt",
            "vanilla_pred",
            "random_pred",
            "apet_pred",
        ]
    ].to_string(index=False)
)

print("\nSaved to:")
print(OUT)
