from pathlib import Path
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "B. Disease Grading"
)


print("=" * 60)
print("IDRiD DATASET INSPECTION")
print("=" * 60)

# ---------------------------------------------------------
# 1. Find CSV files
# ---------------------------------------------------------

csv_files = list(DATASET_PATH.rglob("*.csv"))

print(f"\nFound {len(csv_files)} CSV files:\n")

for file in csv_files:
    print(" -", file.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------
# 2. Inspect each CSV
# ---------------------------------------------------------

for csv_file in csv_files:

    print("\n" + "=" * 60)
    print(f"FILE: {csv_file.name}")
    print("=" * 60)

    df = pd.read_csv(csv_file)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)