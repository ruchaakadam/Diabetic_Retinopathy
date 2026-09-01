from pathlib import Path

import pandas as pd
import numpy as np


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUALITY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "quality"
    / "quality_scores.csv"
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


# =========================================================
# Quality thresholds
# =========================================================

GOOD_THRESHOLD = 65
BORDERLINE_THRESHOLD = 45


# =========================================================
# Metric normalization
# =========================================================

def normalize_focus(value):
    """
    Convert focus score into approximately 0-100.

    Based on the observed IDRiD training distribution.
    """

    score = (
        (value - 4.4)
        / (33.7 - 4.4)
    ) * 100

    return np.clip(score, 0, 100)


def normalize_brightness(value):
    """
    Fundus images should have reasonable illumination.

    Extremely dark or extremely bright images
    receive lower scores.
    """

    ideal = 70
    tolerance = 35

    score = 100 - (
        abs(value - ideal)
        / tolerance
    ) * 100

    return np.clip(score, 0, 100)


def normalize_contrast(value):
    """
    Convert contrast into a 0-100 score.
    """

    score = (
        (value - 20)
        / (68 - 20)
    ) * 100

    return np.clip(score, 0, 100)


def normalize_fov(value):
    """
    Fundus images should have a sufficiently large
    visible retinal field.
    """

    score = (
        (value - 0.63)
        / (0.693 - 0.63)
    ) * 100

    return np.clip(score, 0, 100)


# =========================================================
# Overall quality score
# =========================================================

def calculate_quality_score(row):

    focus_score = normalize_focus(
        row["focus"]
    )

    brightness_score = normalize_brightness(
        row["brightness"]
    )

    contrast_score = normalize_contrast(
        row["contrast"]
    )

    fov_score = normalize_fov(
        row["fov"]
    )

    # Weighted score
    #
    # Focus is important because blurry images
    # can hide retinal lesions.
    #
    # FOV ensures enough retinal area is visible.
    #
    # Brightness and contrast contribute to visibility.

    quality_score = (
        0.35 * focus_score
        + 0.20 * brightness_score
        + 0.25 * contrast_score
        + 0.20 * fov_score
    )

    return float(
        np.clip(
            quality_score,
            0,
            100
        )
    )


# =========================================================
# Quality classification
# =========================================================

def classify_quality(score):

    if score >= GOOD_THRESHOLD:
        return "GOOD"

    elif score >= BORDERLINE_THRESHOLD:
        return "BORDERLINE"

    else:
        return "UNGRADEABLE"


# =========================================================
# Recommended action
# =========================================================

def recommended_action(status):

    if status == "GOOD":

        return "Accept image for classification"

    elif status == "BORDERLINE":

        return "Apply enhancement and re-check"

    else:

        return "Recapture fundus image"


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("IDRiD QUALITY DECISION ENGINE")
    print("=" * 70)

    if not QUALITY_FILE.exists():

        print(
            "\nERROR:"
            "\nquality_scores.csv was not found."
            "\nRun analyze_quality_distribution.py first."
        )

        return

    df = pd.read_csv(
        QUALITY_FILE
    )

    print(
        f"\nImages analyzed: {len(df)}"
    )

    # -----------------------------------------------------
    # Calculate quality scores
    # -----------------------------------------------------

    df["quality_score"] = df.apply(
        calculate_quality_score,
        axis=1
    )

    # -----------------------------------------------------
    # Assign quality status
    # -----------------------------------------------------

    df["quality_status"] = (
        df["quality_score"]
        .apply(classify_quality)
    )

    # -----------------------------------------------------
    # Recommended action
    # -----------------------------------------------------

    df["recommended_action"] = (
        df["quality_status"]
        .apply(recommended_action)
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nQuality status distribution:")

    print(
        df["quality_status"]
        .value_counts()
    )

    print(
        "\nAverage quality score:"
    )

    print(
        f"{df['quality_score'].mean():.2f}/100"
    )

    # -----------------------------------------------------
    # Examples
    # -----------------------------------------------------

    print("\nExample results:")

    print(
        df[
            [
                "image",
                "quality_score",
                "quality_status",
                "recommended_action"
            ]
        ].head(10).to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Save decisions
    # -----------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "quality_decisions.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()