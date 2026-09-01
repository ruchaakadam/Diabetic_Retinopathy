from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "B. Disease Grading"
)

TRAIN_IMAGES = (
    DATASET_PATH
    / "1. Original Images"
    / "a. Training Set"
)

GROUNDTRUTH_PATH = (
    DATASET_PATH
    / "2. Groundtruths"
)

TRAIN_LABELS = (
    GROUNDTRUTH_PATH
    / "a. IDRiD_Disease Grading_Training Labels.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Load labels
# =========================================================

df = pd.read_csv(
    TRAIN_LABELS
)

df = df[
    [
        "Image name",
        "Retinopathy grade"
    ]
].copy()


# =========================================================
# Find images
# =========================================================

image_lookup = {}

for image_path in TRAIN_IMAGES.rglob("*"):

    if image_path.suffix.lower() in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:

        image_lookup[
            image_path.stem
        ] = image_path


# =========================================================
# Attach image paths
# =========================================================

df["image_path"] = df[
    "Image name"
].map(
    lambda name: str(
        image_lookup.get(name)
    )
    if name in image_lookup
    else None
)


# =========================================================
# Check missing images
# =========================================================

missing = df[
    df["image_path"].isna()
]

if len(missing) > 0:

    print(
        "WARNING: Images missing:"
    )

    print(
        missing[
            "Image name"
        ].to_string(index=False)
    )

    raise FileNotFoundError(
        f"{len(missing)} images could not be found."
    )


# =========================================================
# Rename columns
# =========================================================

df = df[
    [
        "image_path",
        "Retinopathy grade"
    ]
].rename(
    columns={
        "Retinopathy grade": "grade"
    }
)


# =========================================================
# Stratified train/validation split
# =========================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["grade"]
)


# =========================================================
# Save
# =========================================================

train_file = (
    SPLIT_DIR
    / "train.csv"
)

val_file = (
    SPLIT_DIR
    / "validation.csv"
)

train_df.to_csv(
    train_file,
    index=False
)

val_df.to_csv(
    val_file,
    index=False
)


# =========================================================
# Report
# =========================================================

print("=" * 70)
print("IDRiD CLASSIFICATION DATASET SPLIT")
print("=" * 70)

print(
    f"\nTotal training-set images: {len(df)}"
)

print(
    f"Model training images:     {len(train_df)}"
)

print(
    f"Validation images:         {len(val_df)}"
)

print(
    "\nTraining distribution:"
)

print(
    train_df["grade"]
    .value_counts()
    .sort_index()
)

print(
    "\nValidation distribution:"
)

print(
    val_df["grade"]
    .value_counts()
    .sort_index()
)

print(
    "\nSaved:"
)

print(train_file)
print(val_file)

print(
    "\nOfficial IDRiD test set remains untouched:"
)

print(
    "103 images"
)