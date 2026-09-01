from pathlib import Path

import cv2
import numpy as np


def load_image(image_path):
    """
    Load a fundus image and convert it to RGB.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def calculate_focus_score(image):
    """
    Estimate image sharpness using variance of Laplacian.

    Higher value = sharper image.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return float(laplacian.var())


def calculate_brightness_score(image):
    """
    Estimate average illumination.

    A fundus image should not be extremely dark
    or extremely bright.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    brightness = float(np.mean(gray))

    return brightness


def calculate_contrast_score(image):
    """
    Estimate contrast using grayscale standard deviation.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    contrast = float(np.std(gray))

    return contrast


def calculate_fov_score(image):
    """
    Estimate the visible retinal field of view.

    Fundus photographs generally contain a roughly
    circular retinal region surrounded by dark pixels.

    This is a baseline heuristic, not a clinical FOV model.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Threshold dark background
    mask = gray > 20

    total_pixels = mask.size
    retinal_pixels = np.sum(mask)

    fov_ratio = retinal_pixels / total_pixels

    return float(fov_ratio)


def analyze_image_quality(image_path):
    """
    Run all quality measurements.
    """

    image = load_image(image_path)

    focus = calculate_focus_score(image)
    brightness = calculate_brightness_score(image)
    contrast = calculate_contrast_score(image)
    fov = calculate_fov_score(image)

    return {
        "focus": focus,
        "brightness": brightness,
        "contrast": contrast,
        "fov": fov
    }