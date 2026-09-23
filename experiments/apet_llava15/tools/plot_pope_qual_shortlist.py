from pathlib import Path
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


EXP = Path("experiments/apet_llava15")
CASE_DIR = EXP / "cases/pope"
IMAGE_DIR = Path("data/eval/pope/val2014")

csv_path = CASE_DIR / "pope_shortlist.csv"
df = pd.read_csv(csv_path)

# 8 cases -> 4 rows x 2 columns
ncols = 2
nrows = (len(df) + ncols - 1) // ncols

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(14, 5.2 * nrows)
)

axes = axes.flatten()

for ax, (_, r) in zip(axes, df.iterrows()):

    image_path = IMAGE_DIR / str(r["image"])

    img = Image.open(image_path).convert("RGB")
    ax.imshow(img)
    ax.axis("off")

    question = str(r["question"]).split(
        "\nAnswer the question"
    )[0]

    question = "\n".join(
        textwrap.wrap(question, width=52)
    )

    title = (
        f"Case {r['case_type']} | {r['category']} | "
        f"QID={r['question_id']}\n"
        f"{question}\n"
        f"GT={r['gt']} | "
        f"Vanilla={r['vanilla_pred']} | "
        f"Random={r['random_pred']} | "
        f"ApET={r['apet_pred']}"
    )

    ax.set_title(
        title,
        fontsize=10,
        pad=8
    )

for ax in axes[len(df):]:
    ax.axis("off")

plt.tight_layout()

png_path = CASE_DIR / "pope_shortlist_contact_sheet.png"
pdf_path = CASE_DIR / "pope_shortlist_contact_sheet.pdf"

plt.savefig(
    png_path,
    dpi=220,
    bbox_inches="tight"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(png_path)
print(pdf_path)
