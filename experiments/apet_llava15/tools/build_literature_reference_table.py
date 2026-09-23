from pathlib import Path
import pandas as pd

EXP = Path("experiments/apet_llava15")
METRICS = EXP / "metrics"

rows = [
    {
        "Method": "ToMe",
        "Avg Visual Tokens": 64,
        "GQA (%)": 48.6,
        "POPE (%)": 52.5,
        "Source": "ApET Table 1 (reported)"
    },
    {
        "Method": "FastV",
        "Avg Visual Tokens": 64,
        "GQA (%)": 46.1,
        "POPE (%)": 48.0,
        "Source": "ApET Table 1 (reported)"
    },
    {
        "Method": "SparseVLM",
        "Avg Visual Tokens": 64,
        "GQA (%)": 52.7,
        "POPE (%)": 75.1,
        "Source": "ApET Table 1 (reported)"
    },
    {
        "Method": "PDrop",
        "Avg Visual Tokens": 64,
        "GQA (%)": 47.5,
        "POPE (%)": 55.9,
        "Source": "ApET Table 1 (reported)"
    },
    {
        "Method": "VisionZip",
        "Avg Visual Tokens": 64,
        "GQA (%)": 55.1,
        "POPE (%)": 77.0,
        "Source": "ApET Table 1 (reported)"
    },
    {
        "Method": "ApET",
        "Avg Visual Tokens": 64,
        "GQA (%)": 56.9,
        "POPE (%)": 84.4,
        "Source": "ApET Table 1 (reported)"
    },
]

out = pd.DataFrame(rows)

path = METRICS / "literature_reported_64tokens.csv"
out.to_csv(path, index=False)

print(out.to_string(index=False))
print("\nSaved:", path)
