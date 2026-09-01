from pathlib import Path
import sys

import cv2
import pandas as pd


# =========================================================
# Make src/ available for imports
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =========================================================
# Project imports
# =========================================================

from quality.image_quality import analyze_image_quality
from quality.quality_decision import (
    calculate_quality_score,
    classify_quality,
)
from preprocessing.enhance_image import enhance_fundus_image


# =========================================================
# Paths
# =========================================================

QUALITY_DECISIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "quality"
    / "quality_decisions.csv"
)

IMAGE_DIR = (
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

ENHANCED_DIR = (
    OUTPUT_DIR
    / "enhanced"
)

ENHANCED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Enhancement + re-check
# =========================================================

def process_borderline_image(image_name):

    original_path = IMAGE_DIR / image_name

    if not original_path.exists():

        return {
            "image": image_name,
            "error": "Original image not found"
        }

    # -----------------------------------------------------
    # Analyze original
    # -----------------------------------------------------

    original_scores = analyze_image_quality(
        original_path
    )

    original_quality = calculate_quality_score(
        original_scores
    )

    # -----------------------------------------------------
    # Load image as BGR
    #
    # enhance_fundus_image() expects BGR.
    # -----------------------------------------------------

    image = cv2.imread(
        str(original_path)
    )

    if image is None:

        return {
            "image": image_name,
            "error": "Could not load image"
        }

    # -----------------------------------------------------
    # Enhance
    # -----------------------------------------------------

    enhanced_image = enhance_fundus_image(
        image
    )

    # -----------------------------------------------------
    # Save enhanced image
    # -----------------------------------------------------

    enhanced_path = (
        ENHANCED_DIR
        / image_name
    )

    cv2.imwrite(
        str(enhanced_path),
        enhanced_image
    )

    # -----------------------------------------------------
    # Re-analyze enhanced image
    # -----------------------------------------------------

    enhanced_scores = analyze_image_quality(
        enhanced_path
    )

    enhanced_quality = calculate_quality_score(
        enhanced_scores
    )

    enhanced_status = classify_quality(
        enhanced_quality
    )

    # -----------------------------------------------------
    # Decide whether enhancement helped
    # -----------------------------------------------------

    improvement = (
        enhanced_quality
        - original_quality
    )

    if enhanced_quality > original_quality:

        final_source = "ENHANCED"

    else:

        final_source = "ORIGINAL"

    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return {
        "image": image_name,

        "original_quality": round(
            original_quality,
            2
        ),

        "enhanced_quality": round(
            enhanced_quality,
            2
        ),

        "improvement": round(
            improvement,
            2
        ),

        "enhanced_status": enhanced_status,

        "final_source": final_source,

        "enhanced_image": (
            str(enhanced_path)
            if final_source == "ENHANCED"
            else ""
        )
    }


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("ADAPTIVE FUNDUS IMAGE QUALITY PIPELINE")
    print("=" * 70)

    # -----------------------------------------------------
    # Load previous quality decisions
    # -----------------------------------------------------

    if not QUALITY_DECISIONS_FILE.exists():

        print(
            "\nERROR:"
            "\nquality_decisions.csv not found."
            "\nRun this first:"
            "\npython src/quality/quality_decision.py"
        )

        return

    df = pd.read_csv(
        QUALITY_DECISIONS_FILE
    )

    print(
        f"\nTotal images: {len(df)}"
    )

    # -----------------------------------------------------
    # Select only borderline images
    # -----------------------------------------------------

    borderline = df[
        df["quality_status"] == "BORDERLINE"
    ].copy()

    print(
        f"Borderline images requiring enhancement: "
        f"{len(borderline)}"
    )

    # -----------------------------------------------------
    # Process borderline images
    # -----------------------------------------------------

    results = []

    for index, row in borderline.iterrows():

        image_name = row["image"]

        print(
            f"\n[{len(results) + 1}/{len(borderline)}] "
            f"{image_name}"
        )

        try:

            result = process_borderline_image(
                image_name
            )

            results.append(
                result
            )

            if "error" not in result:

                print(
                    f"Original score : "
                    f"{result['original_quality']:.2f}"
                )

                print(
                    f"Enhanced score: "
                    f"{result['enhanced_quality']:.2f}"
                )

                print(
                    f"Improvement    : "
                    f"{result['improvement']:+.2f}"
                )

                print(
                    f"Enhanced status: "
                    f"{result['enhanced_status']}"
                )

                print(
                    f"Final source   : "
                    f"{result['final_source']}"
                )

        except Exception as error:

            print(
                f"ERROR processing "
                f"{image_name}: {error}"
            )

            results.append(
                {
                    "image": image_name,
                    "error": str(error)
                }
            )

    # -----------------------------------------------------
    # Create dataframe
    # -----------------------------------------------------

    adaptive_df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # Save adaptive results
    # -----------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "adaptive_quality_results.csv"
    )

    adaptive_df.to_csv(
        output_file,
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    successful = adaptive_df[
        adaptive_df["error"].isna()
    ] if "error" in adaptive_df.columns else adaptive_df

    if len(successful) > 0:

        enhanced_count = (
            successful["final_source"]
            == "ENHANCED"
        ).sum()

        original_count = (
            successful["final_source"]
            == "ORIGINAL"
        ).sum()

        improved_count = (
            successful["improvement"]
            > 0
        ).sum()

        print("\n" + "=" * 70)
        print("ADAPTIVE QUALITY SUMMARY")
        print("=" * 70)

        print(
            f"\nBorderline images processed: "
            f"{len(successful)}"
        )

        print(
            f"Images improved: "
            f"{improved_count}"
        )

        print(
            f"Enhanced image selected: "
            f"{enhanced_count}"
        )

        print(
            f"Original image retained: "
            f"{original_count}"
        )

    print(
        f"\nSaved results to:\n"
        f"{output_file}"
    )

    print(
        f"\nEnhanced images saved to:\n"
        f"{ENHANCED_DIR}"
    )


if __name__ == "__main__":
    main()