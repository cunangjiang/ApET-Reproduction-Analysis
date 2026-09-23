from pathlib import Path
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


EXP = Path("experiments/apet_llava15")

POOL = (
    EXP /
    "cases/pope/pope_case_pool.csv"
)

IMAGE_DIR = Path(
    "data/eval/pope/val2014"
)

OUT_DIR = (
    EXP /
    "figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Final manually selected cases
# --------------------------------------------------
case_info = [
    (
        20002561,
        "Small / peripheral object",
    ),
    (
        20002309,
        "Cluttered scene",
    ),
    (
        10000553,
        "Scene coverage",
    ),
    (
        10002785,
        "ApET-only success",
    ),
    (
        2372,
        "Compression failure",
    ),
    (
        10001105,
        "ApET failure",
    ),
]


df = pd.read_csv(POOL)

# Avoid dtype mismatch
df["question_id"] = (
    df["question_id"]
    .astype(str)
)

case_ids = [
    str(x[0])
    for x in case_info
]

case_labels = {
    str(qid): label
    for qid, label in case_info
}

selected = (
    df[
        df["question_id"].isin(case_ids)
    ]
    .copy()
)

selected["order"] = (
    selected["question_id"]
    .map({
        qid: i
        for i, qid in enumerate(case_ids)
    })
)

selected = (
    selected
    .sort_values("order")
    .reset_index(drop=True)
)

assert len(selected) == 6


fig, axes = plt.subplots(
    2,
    3,
    figsize=(15.5, 9.8)
)

axes = axes.flatten()


for i, (_, r) in enumerate(
    selected.iterrows()
):

    ax = axes[i]

    img_path = (
        IMAGE_DIR /
        str(r["image"])
    )

    img = Image.open(
        img_path
    ).convert("RGB")

    ax.imshow(img)
    ax.axis("off")

    q = str(r["question"]).split(
        "\nAnswer the question"
    )[0]

    q = "\n".join(
        textwrap.wrap(
            q,
            width=38
        )
    )

    label = case_labels[
        str(r["question_id"])
    ]

    title = (
        f"{label}\n"
        f"{q}\n"
        f"GT: {str(r['gt']).capitalize()}   |   "
        f"Vanilla: {str(r['vanilla_pred']).capitalize()}\n"
        f"RandomPrune: {str(r['random_pred']).capitalize()}   |   "
        f"ApET: {str(r['apet_pred']).capitalize()}"
    )

    ax.set_title(
        title,
        fontsize=10,
        pad=8
    )


plt.tight_layout(
    h_pad=2.2,
    w_pad=1.5
)

png = (
    OUT_DIR /
    "pope_qualitative_cases.png"
)

pdf = (
    OUT_DIR /
    "pope_qualitative_cases.pdf"
)

plt.savefig(
    png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf,
    bbox_inches="tight"
)

plt.close()


selected[
    [
        "question_id",
        "category",
        "image",
        "question",
        "gt",
        "vanilla_pred",
        "random_pred",
        "apet_pred",
        "pattern",
    ]
].to_csv(
    EXP /
    "cases/pope/final_selected_cases.csv",
    index=False
)


print("Saved:")
print(png)
print(pdf)
print(
    EXP /
    "cases/pope/final_selected_cases.csv"
)
