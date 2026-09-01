from pathlib import Path
import sys

import cv2
import numpy as np


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# IMPORT EXISTING MODULES
# =========================================================

sys.path.append(
    str(PROJECT_ROOT / "src" / "quality")
)

sys.path.append(
    str(PROJECT_ROOT / "src" / "preprocessing")
)

from image_quality import analyze_image_quality


# =========================================================
# QUALITY THRESHOLDS
# =========================================================

GOOD_THRESHOLD = 65

BORDERLINE_THRESHOLD = 45


# =========================================================
# QUALITY NORMALIZATION
# =========================================================

def normalize_focus(value):

    score = (
        (value - 4.4)
        / (33.7 - 4.4)
    ) * 100

    return np.clip(
        score,
        0,
        100
    )


def normalize_brightness(value):

    ideal = 70

    tolerance = 35

    score = 100 - (
        abs(value - ideal)
        / tolerance
    ) * 100

    return np.clip(
        score,
        0,
        100
    )


def normalize_contrast(value):

    score = (
        (value - 20)
        / (68 - 20)
    ) * 100

    return np.clip(
        score,
        0,
        100
    )


def normalize_fov(value):

    score = (
        (value - 0.63)
        / (0.693 - 0.63)
    ) * 100

    return np.clip(
        score,
        0,
        100
    )


def calculate_quality_score(scores):

    focus = normalize_focus(
        scores["focus"]
    )

    brightness = normalize_brightness(
        scores["brightness"]
    )

    contrast = normalize_contrast(
        scores["contrast"]
    )

    fov = normalize_fov(
        scores["fov"]
    )

    score = (
        0.35 * focus
        + 0.20 * brightness
        + 0.25 * contrast
        + 0.20 * fov
    )

    return float(
        np.clip(
            score,
            0,
            100
        )
    )


def classify_quality(score):

    if score >= GOOD_THRESHOLD:

        return "GOOD"

    elif score >= BORDERLINE_THRESHOLD:

        return "BORDERLINE"

    else:

        return "UNGRADEABLE"


# =========================================================
# FUNDUS MASK
# =========================================================

def create_fundus_mask(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        return mask

    largest = max(
        contours,
        key=cv2.contourArea
    )

    clean_mask = np.zeros_like(
        gray
    )

    cv2.drawContours(
        clean_mask,
        [largest],
        -1,
        255,
        thickness=-1
    )

    clean_mask = cv2.GaussianBlur(
        clean_mask,
        (7, 7),
        0
    )

    return clean_mask


# =========================================================
# ILLUMINATION NORMALIZATION
# =========================================================

def normalize_illumination(
    image,
    mask
):

    green = image[
        :,
        :,
        1
    ].astype(
        np.float32
    )

    background = cv2.GaussianBlur(
        green,
        (0, 0),
        sigmaX=30
    )

    background = np.maximum(
        background,
        1.0
    )

    retinal_pixels = (
        mask > 128
    )

    if np.any(
        retinal_pixels
    ):

        target = np.median(
            background[
                retinal_pixels
            ]
        )

    else:

        target = 128.0

    normalized_green = (
        green
        / background
    ) * target

    normalized_green = np.clip(
        normalized_green,
        0,
        255
    ).astype(
        np.uint8
    )

    result = image.copy()

    result[
        :,
        :,
        1
    ] = np.where(
        retinal_pixels,
        normalized_green,
        green.astype(
            np.uint8
        )
    )

    return result


# =========================================================
# CLAHE
# =========================================================

def apply_clahe(
    image,
    mask
):

    green = image[
        :,
        :,
        1
    ]

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    enhanced_green = clahe.apply(
        green
    )

    retinal_pixels = (
        mask > 128
    )

    result = image.copy()

    result[
        :,
        :,
        1
    ] = np.where(
        retinal_pixels,
        enhanced_green,
        green
    )

    return result


# =========================================================
# DENOISING
# =========================================================

def denoise_image(
    image,
    mask
):

    denoised = cv2.GaussianBlur(
        image,
        (3, 3),
        0.3
    )

    retinal_pixels = (
        mask > 128
    )

    result = image.copy()

    for channel in range(3):

        result[
            :,
            :,
            channel
        ] = np.where(
            retinal_pixels,
            denoised[
                :,
                :,
                channel
            ],
            image[
                :,
                :,
                channel
            ]
        )

    return result


# =========================================================
# COMPLETE ENHANCEMENT
# =========================================================

def enhance_fundus_image(
    image
):

    mask = create_fundus_mask(
        image
    )

    normalized = (
        normalize_illumination(
            image,
            mask
        )
    )

    enhanced = apply_clahe(
        normalized,
        mask
    )

    result = denoise_image(
        enhanced,
        mask
    )

    return result


# =========================================================
# SAVE IMAGE
# =========================================================

def save_image(
    image,
    path
):

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(path),
        image_bgr
    )


# =========================================================
# TEST / COMPARE
# =========================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python src/inference/"
            "enhanced_pipeline.py "
            "<image_path>"
        )

        return


    image_path = Path(
        sys.argv[1]
    )


    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )


    print(
        "=" * 70
    )

    print(
        "FUNDUS ENHANCEMENT + QUALITY RE-CHECK"
    )

    print(
        "=" * 70
    )


    # -----------------------------------------------------
    # Original
    # -----------------------------------------------------

    original_bgr = cv2.imread(
        str(image_path)
    )

    if original_bgr is None:

        raise ValueError(
            "Could not load image."
        )


    original = cv2.cvtColor(
        original_bgr,
        cv2.COLOR_BGR2RGB
    )


    original_metrics = (
        analyze_image_quality(
            image_path
        )
    )


    original_score = (
        calculate_quality_score(
            original_metrics
        )
    )


    original_status = (
        classify_quality(
            original_score
        )
    )


    print(
        "\nORIGINAL IMAGE"
    )

    print(
        f"Quality score: "
        f"{original_score:.2f}/100"
    )

    print(
        f"Status: "
        f"{original_status}"
    )


    # -----------------------------------------------------
    # Enhancement
    # -----------------------------------------------------

    enhanced = enhance_fundus_image(
        original
    )


    # -----------------------------------------------------
    # Save temporary enhanced image
    # -----------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "quality"
        / "enhanced"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    enhanced_path = (
        output_dir
        / f"{image_path.stem}_enhanced.jpg"
    )


    save_image(
        enhanced,
        enhanced_path
    )


    # -----------------------------------------------------
    # Re-analyze enhanced image
    # -----------------------------------------------------

    enhanced_metrics = (
        analyze_image_quality(
            enhanced_path
        )
    )


    enhanced_score = (
        calculate_quality_score(
            enhanced_metrics
        )
    )


    enhanced_status = (
        classify_quality(
            enhanced_score
        )
    )


    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    improvement = (
        enhanced_score
        - original_score
    )


    print(
        "\nENHANCED IMAGE"
    )

    print(
        f"Quality score: "
        f"{enhanced_score:.2f}/100"
    )

    print(
        f"Status: "
        f"{enhanced_status}"
    )


    print(
        "\nQUALITY CHANGE"
    )

    print(
        f"Score change: "
        f"{improvement:+.2f}"
    )


    if improvement > 0:

        print(
            "Result: Quality score improved."
        )

    elif improvement < 0:

        print(
            "Result: Quality score decreased."
        )

    else:

        print(
            "Result: No quality-score change."
        )


    print(
        "\nEnhanced image saved:"
    )

    print(
        enhanced_path
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "ENHANCEMENT TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()