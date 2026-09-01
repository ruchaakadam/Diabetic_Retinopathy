from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


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

TRAIN_IMAGES = (
    DATASET_PATH
    / "1. Original Images"
    / "a. Training Set"
)

GROUNDTRUTH_PATH = DATASET_PATH / "2. Groundtruths"

TRAIN_CSV = (
    GROUNDTRUTH_PATH
    / "a. IDRiD_Disease Grading_Training Labels.csv"
)


# ---------------------------------------------------------
# Load labels
# ---------------------------------------------------------

df = pd.read_csv(TRAIN_CSV)

df = df[
    [
        "Image name",
        "Retinopathy grade"
    ]
]


# ---------------------------------------------------------
# Find image files
# ---------------------------------------------------------

image_files = {}

for image_path in TRAIN_IMAGES.rglob("*"):

    if image_path.suffix.lower() in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:

        image_files[image_path.stem] = image_path


print(f"Found {len(image_files)} training images.")


# ---------------------------------------------------------
# Select one image from each grade
# ---------------------------------------------------------

selected = []

for grade in range(5):

    grade_rows = df[
        df["Retinopathy grade"] == grade
    ]

    if len(grade_rows) == 0:
        continue

    row = grade_rows.iloc[0]

    image_name = row["Image name"]

    image_path = image_files.get(image_name)

    if image_path is not None:

        selected.append(
            (
                grade,
                image_path
            )
        )


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    len(selected),
    figsize=(20, 5)
)

if len(selected) == 1:
    axes = [axes]


grade_names = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}


for ax, (grade, image_path) in zip(
    axes,
    selected
):

    image = Image.open(image_path)

    ax.imshow(image)

    ax.set_title(
        f"Grade {grade}\n{grade_names[grade]}"
    )

    ax.axis("off")


plt.tight_layout()

plt.show()