from pathlib import Path
import pandas as pd

from quality.image_quality import analyze_image_quality
from preprocessing.enhance_image import enhance_fundus_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


GOOD_THRESHOLD = 65
BORDERLINE_THRESHOLD = 45


def classify_quality(score):

    if score >= GOOD_THRESHOLD:
        return "GOOD"

    elif score >= BORDERLINE_THRESHOLD:
        return "BORDERLINE"

    else:
        return "UNGRADEABLE"


def get_action(status):

    if status == "GOOD":
        return "ACCEPT"

    elif status == "BORDERLINE":
        return "ENHANCE_AND_RECHECK"

    else:
        return "RECAPTURE"


def main():

    print("=" * 70)
    print("ADAPTIVE IMAGE QUALITY PIPELINE")
    print("=" * 70)

    image_files = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    print(
        f"\nFound {len(image_files)} images"
    )

    results = []

    for image_path in image_files:

        scores = analyze_image_quality(
            str(image_path)
        )

        quality_score = (
            scores["quality_score"]
            if "quality_score" in scores
            else 0
        )

        status = classify_quality(
            quality_score
        )

        action = get_action(
            status
        )

        results.append({
            "image": image_path.name,
            "quality_score": quality_score,
            "initial_status": status,
            "action": action
        })

    df = pd.DataFrame(results)

    output_file = (
        OUTPUT_DIR
        / "adaptive_quality_results.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("\nQUALITY DISTRIBUTION")
    print(
        df["initial_status"]
        .value_counts()
    )

    print(
        f"\nSaved results to:\n{output_file}"
    )


if __name__ == "__main__":
    main()