from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "B. Disease Grading"
)

GROUNDTRUTH_PATH = DATASET_PATH / "2. Groundtruths"


TRAIN_CSV = (
    GROUNDTRUTH_PATH
    / "a. IDRiD_Disease Grading_Training Labels.csv"
)

TEST_CSV = (
    GROUNDTRUTH_PATH
    / "b. IDRiD_Disease Grading_Testing Labels.csv"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)


# Keep only the columns we need

columns = [
    "Image name",
    "Retinopathy grade",
    "Risk of macular edema "
]

train_df = train_df[columns]
test_df = test_df[columns]


# Combine for overall analysis

all_df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)


# ---------------------------------------------------------
# Dataset information
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("IDRiD LABEL ANALYSIS")
print("=" * 60)

print(f"\nTraining images: {len(train_df)}")
print(f"Testing images:  {len(test_df)}")
print(f"Total images:    {len(all_df)}")


# ---------------------------------------------------------
# DR grade distribution
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RETINOPATHY GRADE DISTRIBUTION")
print("=" * 60)

grade_counts = (
    all_df["Retinopathy grade"]
    .value_counts()
    .sort_index()
)

print(grade_counts)


# ---------------------------------------------------------
# Percentages
# ---------------------------------------------------------

print("\nGrade percentages:")

grade_percentages = (
    all_df["Retinopathy grade"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

for grade, percentage in grade_percentages.items():

    print(
        f"Grade {grade}: "
        f"{percentage:.2f}%"
    )


# ---------------------------------------------------------
# Macular edema distribution
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MACULAR EDEMA RISK DISTRIBUTION")
print("=" * 60)

print(
    all_df["Risk of macular edema "]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# Plot DR distribution
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

grade_counts.plot(
    kind="bar"
)

plt.title(
    "IDRiD Diabetic Retinopathy Grade Distribution"
)

plt.xlabel(
    "Retinopathy Grade"
)

plt.ylabel(
    "Number of Images"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.show()