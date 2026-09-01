from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from image_quality import analyze_image_quality


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "B. Disease Grading"
    / "1. Original Images"
    / "a. Training Set"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "quality"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Find training images
# ---------------------------------------------------------

image_files = []

for path in TRAIN_IMAGES.rglob("*"):

    if path.suffix.lower() in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:
        image_files.append(path)


print("=" * 70)
print("IDRiD IMAGE QUALITY ANALYSIS")
print("=" * 70)

print(
    f"\nFound {len(image_files)} training images."
)


# ---------------------------------------------------------
# Analyze every image
# ---------------------------------------------------------

results = []

for index, image_path in enumerate(image_files):

    try:

        scores = analyze_image_quality(
            image_path
        )

        results.append(
            {
                "image": image_path.name,
                "focus": scores["focus"],
                "brightness": scores["brightness"],
                "contrast": scores["contrast"],
                "fov": scores["fov"]
            }
        )

    except Exception as error:

        print(
            f"Could not process "
            f"{image_path.name}: {error}"
        )

    if (index + 1) % 50 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(image_files)}"
        )


# ---------------------------------------------------------
# Create dataframe
# ---------------------------------------------------------

df = pd.DataFrame(results)


print("\nAnalysis complete.")

print(
    f"Successfully analyzed "
    f"{len(df)} images."
)


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("QUALITY SCORE STATISTICS")
print("=" * 70)

print(
    df[
        [
            "focus",
            "brightness",
            "contrast",
            "fov"
        ]
    ].describe()
)


# ---------------------------------------------------------
# Save numerical results
# ---------------------------------------------------------

output_file = (
    OUTPUT_DIR
    / "quality_scores.csv"
)

df.to_csv(
    output_file,
    index=False
)

print(
    f"\nSaved results to:\n"
    f"{output_file}"
)


# ---------------------------------------------------------
# Create distributions
# ---------------------------------------------------------

metrics = [
    "focus",
    "brightness",
    "contrast",
    "fov"
]

for metric in metrics:

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        df[metric],
        bins=30
    )

    plt.title(
        f"IDRiD {metric.capitalize()} Distribution"
    )

    plt.xlabel(metric)

    plt.ylabel(
        "Number of Images"
    )

    plt.tight_layout()

    output_plot = (
        OUTPUT_DIR
        / f"{metric}_distribution.png"
    )

    plt.savefig(
        output_plot,
        dpi=150
    )

    plt.show()