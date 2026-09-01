from pathlib import Path
import sys
import csv


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from inference.predict import run_pipeline


# =========================================================
# DATASET PATHS
# =========================================================

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "B. Disease Grading"
    / "1. Original Images"
    / "b. Testing Set"
)

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "B. Disease Grading"
    / "2. Groundtruths"
    / "b. IDRiD_Disease Grading_Testing Labels.csv"
)


# =========================================================
# OUTPUT
# =========================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "normal_validation_results.csv"
)


# =========================================================
# LOAD NORMAL IMAGE LABELS
# =========================================================

normal_images = []

with open(
    LABEL_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        image_name = (
            row["Image name"]
            .strip()
        )

        grade = int(
            row["Retinopathy grade"]
        )

        # Grade 0 = normal
        if grade == 0:

            normal_images.append(
                image_name
            )


# =========================================================
# COUNTERS
# =========================================================

total_normal = len(
    normal_images
)

correct = 0
incorrect = 0
errors = 0
ungradable = 0

rows = []


# =========================================================
# VALIDATION
# =========================================================

print()
print("=" * 60)
print("NORMAL IMAGE VALIDATION")
print("=" * 60)

print(
    f"Normal images found: {total_normal}"
)

print()


for index, image_name in enumerate(
    normal_images,
    start=1
):

    image_path = (
        IMAGE_ROOT
        / f"{image_name}.jpg"
    )

    print()
    print("=" * 60)
    print(
        f"Testing {index}/{total_normal}: "
        f"{image_name}"
    )
    print("=" * 60)

    # -----------------------------------------------------
    # CHECK IMAGE EXISTS
    # -----------------------------------------------------

    if not image_path.exists():

        errors += 1

        print(
            f"ERROR: Image not found: "
            f"{image_path}"
        )

        rows.append({

            "image":
                image_name,

            "quality":
                "",

            "classification_source":
                "",

            "probability":
                "",

            "probability_percent":
                "",

            "decision":
                "",

            "correct":
                False,

            "error":
                "Image file not found"

        })

        continue


    # -----------------------------------------------------
    # RUN PIPELINE
    # -----------------------------------------------------

    try:

        result = run_pipeline(
            image_path
        )


        # =================================================
        # UNGRADEABLE IMAGE
        # =================================================

        if result is None:

            ungradable += 1

            print()
            print(
                "Quality: UNGRADEABLE"
            )

            print(
                "Classification: "
                "NOT PERFORMED"
            )

            print(
                "Correct: Not applicable"
            )

            rows.append({

                "image":
                    image_name,

                "quality":
                    "UNGRADEABLE",

                "classification_source":
                    "",

                "probability":
                    "",

                "probability_percent":
                    "",

                "decision":
                    "UNGRADEABLE",

                "correct":
                    "",

                "error":
                    ""

            })

            continue


        # =================================================
        # READ PIPELINE RESULT
        # =================================================

        quality = result.get(
            "final_quality_status",
            result.get(
                "quality_status",
                ""
            )
        )

        classification_source = result.get(
            "classification_source",
            ""
        )

        probability = float(
            result.get(
                "referable_probability",
                0
            )
        )

        probability_percent = (
            probability * 100
        )

        decision = result.get(
            "decision",
            ""
        )


        # =================================================
        # DETERMINE CORRECTNESS
        # =================================================

        is_correct = (
            decision
            == "NON-REFERABLE DR"
        )


        if is_correct:

            correct += 1

        else:

            incorrect += 1


        # =================================================
        # PRINT RESULT
        # =================================================

        print()

        print(
            f"Quality: {quality}"
        )

        print(
            f"Source: "
            f"{classification_source}"
        )

        print(
            f"Probability: "
            f"{probability:.4f}"
        )

        print(
            f"Decision: {decision}"
        )

        print(
            f"Correct: {is_correct}"
        )


        # =================================================
        # SAVE RESULT
        # =================================================

        rows.append({

            "image":
                image_name,

            "quality":
                quality,

            "classification_source":
                classification_source,

            "probability":
                probability,

            "probability_percent":
                probability_percent,

            "decision":
                decision,

            "correct":
                is_correct,

            "error":
                ""

        })


    # =====================================================
    # PIPELINE ERROR
    # =====================================================

    except Exception as error:

        errors += 1

        print()

        print(
            f"ERROR: {error}"
        )

        rows.append({

            "image":
                image_name,

            "quality":
                "",

            "classification_source":
                "",

            "probability":
                "",

            "probability_percent":
                "",

            "decision":
                "",

            "correct":
                False,

            "error":
                str(error)

        })


# =========================================================
# ACCURACY
# =========================================================

tested = (
    correct
    + incorrect
)

accuracy = (

    correct / tested * 100

    if tested > 0

    else 0

)


# =========================================================
# SAVE CSV
# =========================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    fieldnames = [

        "image",
        "quality",
        "classification_source",
        "probability",
        "probability_percent",
        "decision",
        "correct",
        "error"

    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 60)
print("NORMAL IMAGE VALIDATION COMPLETE")
print("=" * 60)

print(
    f"Total normal images: "
    f"{total_normal}"
)

print(
    f"Successfully tested: "
    f"{tested}"
)

print(
    f"Correct: "
    f"{correct}"
)

print(
    f"Incorrect: "
    f"{incorrect}"
)

print(
    f"Errors: "
    f"{errors}"
)

print(
    f"Ungradeable: "
    f"{ungradable}"
)

print(
    f"Normal-image accuracy: "
    f"{accuracy:.2f}%"
)

print()

print(
    f"Results saved to:"
)

print(
    OUTPUT_FILE
)

print("=" * 60)