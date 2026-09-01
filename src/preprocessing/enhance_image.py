import cv2
import numpy as np


def create_fundus_mask(image):
    """
    Detect the circular retinal field and create a mask.
    Everything outside the retinal field remains unchanged.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Threshold dark background
    _, mask = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # Keep the largest connected region
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

    clean_mask = np.zeros_like(gray)

    cv2.drawContours(
        clean_mask,
        [largest],
        -1,
        255,
        thickness=-1
    )

    # Slightly smooth the mask boundary
    clean_mask = cv2.GaussianBlur(
        clean_mask,
        (7, 7),
        0
    )

    return clean_mask


def normalize_illumination(image, mask):
    """
    Conservative illumination correction on the green channel.
    """

    green = image[:, :, 1].astype(
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

    # Only calculate correction inside the retina
    retinal_pixels = (
        mask > 128
    )

    if np.any(retinal_pixels):

        target = np.median(
            background[retinal_pixels]
        )

    else:

        target = 128.0

    normalized_green = (
        green / background
    ) * target

    normalized_green = np.clip(
        normalized_green,
        0,
        255
    ).astype(np.uint8)

    result = image.copy()

    # Apply only inside retinal field
    result[:, :, 1] = np.where(
        retinal_pixels,
        normalized_green,
        green.astype(np.uint8)
    )

    return result


def apply_clahe(image, mask):
    """
    Apply conservative CLAHE only inside the retinal field.
    """

    green = image[:, :, 1]

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

    result[:, :, 1] = np.where(
        retinal_pixels,
        enhanced_green,
        green
    )

    return result


def denoise_image(image, mask):
    """
    Very mild denoising inside the retinal field.
    """

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

        result[:, :, channel] = np.where(
            retinal_pixels,
            denoised[:, :, channel],
            image[:, :, channel]
        )

    return result


def enhance_fundus_image(image):
    """
    Complete conservative fundus enhancement pipeline.
    """

    mask = create_fundus_mask(
        image
    )

    normalized = normalize_illumination(
        image,
        mask
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