from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from enhance_image import enhance_fundus_image


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "B. Disease Grading"
    / "1. Original Images"
    / "a. Training Set"
    / "IDRiD_118.jpg"
)


# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        f"Could not load:\n{IMAGE_PATH}"
    )


# OpenCV uses BGR.
# Matplotlib expects RGB.
original_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ---------------------------------------------------------
# Enhancement
# ---------------------------------------------------------

enhanced = enhance_fundus_image(
    image
)

enhanced_rgb = cv2.cvtColor(
    enhanced,
    cv2.COLOR_BGR2RGB
)


# ---------------------------------------------------------
# Display comparison
# ---------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)

plt.subplot(1, 2, 1)

plt.imshow(
    original_rgb
)

plt.title(
    "Original - IDRiD_118"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    enhanced_rgb
)

plt.title(
    "Enhanced - CLAHE + Illumination Normalization"
)

plt.axis("off")


plt.tight_layout()

plt.show()
