from pathlib import Path
import sys
import json

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# IMPORT PROJECT MODULES
# =========================================================
from inference.report_generator import (
    save_screening_report
)
sys.path.append(
    str(PROJECT_ROOT / "src" / "quality")
)

sys.path.append(
    str(PROJECT_ROOT / "src" / "classification")
)

from image_quality import analyze_image_quality
from model import create_model


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dr_referable_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "explainability"
)

ENHANCED_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "quality"
    / "enhanced"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ENHANCED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIGURATION
# =========================================================

THRESHOLD = 0.55

GOOD_THRESHOLD = 65

BORDERLINE_THRESHOLD = 45


# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# QUALITY NORMALIZATION
# =========================================================

def normalize_focus(value):

    score = (
        (value - 4.4)
        / (33.7 - 4.4)
    ) * 100

    return float(
        np.clip(
            score,
            0,
            100
        )
    )


def normalize_brightness(value):

    ideal = 70

    tolerance = 35

    score = 100 - (
        abs(value - ideal)
        / tolerance
    ) * 100

    return float(
        np.clip(
            score,
            0,
            100
        )
    )


def normalize_contrast(value):

    score = (
        (value - 20)
        / (68 - 20)
    ) * 100

    return float(
        np.clip(
            score,
            0,
            100
        )
    )


def normalize_fov(value):

    score = (
        (value - 0.63)
        / (0.693 - 0.63)
    ) * 100

    return float(
        np.clip(
            score,
            0,
            100
        )
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
# FUNDUS ENHANCEMENT
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
# QUALITY ANALYSIS FROM ARRAY
# =========================================================

def analyze_array_quality(
    image
):

    temporary_path = (
        ENHANCED_DIR
        / "_temporary_quality_check.jpg"
    )

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(temporary_path),
        image_bgr
    )

    metrics = analyze_image_quality(
        temporary_path
    )

    score = calculate_quality_score(
        metrics
    )

    status = classify_quality(
        score
    )

    return (
        metrics,
        score,
        status
    )


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    print(
        "\nLoading referable DR model..."
    )

    model = create_model(
        num_classes=2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    return model


# =========================================================
# CLASSIFICATION
# =========================================================

def predict_image(
    model,
    image
):

    tensor = transform(
        Image.fromarray(
            image
        )
    )

    tensor = tensor.unsqueeze(
        0
    ).to(
        DEVICE
    )

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

    referable_probability = float(
        probabilities[
            0,
            1
        ]
        .cpu()
        .item()
    )

    if (
        referable_probability
        >= THRESHOLD
    ):

        decision = (
            "REFERABLE DR"
        )

    else:

        decision = (
            "NON-REFERABLE DR"
        )

    return (
        referable_probability,
        decision
    )


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(
    model,
    image
):

    activations = None

    gradients = None


    def forward_hook(
        module,
        inputs,
        output
    ):

        nonlocal activations

        activations = output


    target_layer = (
        model.features[-1]
    )

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )


    try:

        tensor = transform(
            Image.fromarray(
                image
            )
        )

        tensor = tensor.unsqueeze(
            0
        ).to(
            DEVICE
        )


        model.zero_grad(
            set_to_none=True
        )


        # IMPORTANT:
        # Do NOT use torch.no_grad()
        # here because Grad-CAM needs gradients.

        with torch.enable_grad():

            output = model(
                tensor
            )


            if activations is None:

                raise RuntimeError(
                    "Grad-CAM activations "
                    "were not captured."
                )


            def activation_gradient_hook(
                grad
            ):

                nonlocal gradients

                gradients = grad


            if not activations.requires_grad:

                raise RuntimeError(
                    "Grad-CAM activation tensor "
                    "does not require gradients."
                )


            activations.register_hook(
                activation_gradient_hook
            )


            probabilities = torch.softmax(
                output,
                dim=1
            )


            predicted_class = int(
                torch.argmax(
                    probabilities,
                    dim=1
                ).item()
            )


            probability = float(
                probabilities[
                    0,
                    predicted_class
                ]
                .detach()
                .cpu()
                .item()
            )


            score = output[
                0,
                predicted_class
            ]


            score.backward()


        if gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients "
                "were not captured."
            )


        acts = activations.detach()

        grads = gradients.detach()


        # -------------------------------------------------
        # Global average pooling
        # -------------------------------------------------

        weights = grads.mean(
            dim=(2, 3),
            keepdim=True
        )


        # -------------------------------------------------
        # Weighted feature maps
        # -------------------------------------------------

        cam = (
            weights * acts
        ).sum(
            dim=1
        )


        cam = torch.relu(
            cam
        )


        cam = cam.squeeze(
            0
        )


        cam = (
            cam
            .detach()
            .cpu()
            .numpy()
        )


        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 0:

            cam /= maximum


        # -------------------------------------------------
        # Resize
        # -------------------------------------------------

        height, width = (
            image.shape[:2]
        )

        cam = cv2.resize(
            cam,
            (width, height)
        )


        # -------------------------------------------------
        # Heatmap
        # -------------------------------------------------

        heatmap = np.uint8(
            cam * 255
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )


        # -------------------------------------------------
        # Overlay
        # -------------------------------------------------

        overlay = (
            0.55 * image
            + 0.45 * heatmap
        )

        overlay = np.clip(
            overlay,
            0,
            255
        ).astype(
            np.uint8
        )


        return (
            heatmap,
            overlay,
            predicted_class,
            probability
        )


    finally:

        forward_handle.remove()


# =========================================================
# SAVE GRAD-CAM
# =========================================================

def save_gradcam(
    image,
    heatmap,
    overlay,
    decision,
    probability,
    output_file
):

    plt.figure(
        figsize=(15, 5)
    )


    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        image
    )

    plt.title(
        "Fundus Image"
    )

    plt.axis(
        "off"
    )


    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        heatmap
    )

    plt.title(
        "Grad-CAM Heatmap"
    )

    plt.axis(
        "off"
    )


    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        overlay
    )

    plt.title(
        f"{decision}\n"
        f"Probability: "
        f"{probability:.2%}"
    )

    plt.axis(
        "off"
    )


    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# SCREENING REPORT
# =========================================================

def create_screening_report(
    image_path,
    original_metrics,
    original_score,
    original_status,
    final_quality_score,
    final_quality_status,
    classification_source,
    referable_probability,
    decision,
    gradcam_file
):

    enhancement_applied = (
        classification_source
        == "ENHANCED"
    )


    report = {

        "project":
            "Diabetic Retinopathy AI Screening",

        "image":
            image_path.name,

        "image_path":
            str(image_path),

        "device":
            str(DEVICE),

        "image_quality": {

            "original": {

                "focus":
                    float(
                        original_metrics[
                            "focus"
                        ]
                    ),

                "brightness":
                    float(
                        original_metrics[
                            "brightness"
                        ]
                    ),

                "contrast":
                    float(
                        original_metrics[
                            "contrast"
                        ]
                    ),

                "fov":
                    float(
                        original_metrics[
                            "fov"
                        ]
                    ),

                "quality_score":
                    round(
                        float(
                            original_score
                        ),
                        2
                    ),

                "status":
                    original_status
            },


            "final": {

                "quality_score":
                    round(
                        float(
                            final_quality_score
                        ),
                        2
                    ),

                "status":
                    final_quality_status
            },


            "enhancement_applied":
                enhancement_applied,

            "quality_improvement":
                round(
                    float(
                        final_quality_score
                        - original_score
                    ),
                    2
                )
        },


        "classification": {

            "source":
                classification_source,

            "referable_probability":
                round(
                    float(
                        referable_probability
                    ),
                    6
                ),

            "threshold":
                THRESHOLD,

            "decision":
                decision
        },


        "explainability": {

            "method":
                "Grad-CAM",

            "generated":
                True,

            "file":
                str(gradcam_file)
        },


        "disclaimer":
            "This is an AI-assisted "
            "screening prototype and "
            "not a clinical diagnosis."
    }


    return report

# =========================================================
# MAIN PIPELINE
# =========================================================

def run_pipeline(
    image_path
):

    image_path = Path(
        image_path
    )


    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )


    print("=" * 70)

    print(
        "DIABETIC RETINOPATHY AI SCREENING"
    )

    print("=" * 70)


    print(
        f"\nImage:"
        f"\n{image_path}"
    )

    print(
        f"\nDevice: {DEVICE}"
    )


    # =====================================================
    # LOAD IMAGE
    # =====================================================

    image_bgr = cv2.imread(
        str(image_path)
    )


    if image_bgr is None:

        raise ValueError(
            "Could not load image."
        )


    original_image = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # ORIGINAL QUALITY
    # =====================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "IMAGE QUALITY"
    )

    print(
        "-" * 70
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
        f"Focus: "
        f"{original_metrics['focus']:.2f}"
    )

    print(
        f"Brightness: "
        f"{original_metrics['brightness']:.2f}"
    )

    print(
        f"Contrast: "
        f"{original_metrics['contrast']:.2f}"
    )

    print(
        f"FOV: "
        f"{original_metrics['fov']:.4f}"
    )

    print(
        f"\nQuality Score: "
        f"{original_score:.2f}/100"
    )

    print(
        f"Quality Status: "
        f"{original_status}"
    )


    # =====================================================
    # SELECT IMAGE FOR CLASSIFICATION
    # =====================================================

    classification_image = (
        original_image
    )

    classification_source = (
        "ORIGINAL"
    )

    final_quality_score = (
        original_score
    )

    final_quality_status = (
        original_status
    )


    # =====================================================
    # UNGRADEABLE
    # =====================================================

    if (
        original_status
        == "UNGRADEABLE"
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            "SCREENING STOPPED"
        )

        print(
            "-" * 70
        )

        print(
            "\nImage is ungradable."
        )

        print(
            "Recommendation:"
        )

        print(
            "Recapture fundus image."
        )

        return {
            "final_quality":
                float(original_score),

            "quality_status":
                "UNGRADEABLE",

            "classification_source":
                None,

            "referable_probability":
                None,

            "decision":
                "IMAGE UNGRADEABLE",

            "gradcam_generated":
                False,

            "gradcam_path":
                None,

            "report_path":
                None,

            "recommendation":
                "Recapture fundus image."
        }

    # =====================================================
    # BORDERLINE → ENHANCEMENT
    # =====================================================

    if (
        original_status
        == "BORDERLINE"
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            "BORDERLINE IMAGE DETECTED"
        )

        print(
            "-" * 70
        )

        print(
            "\nApplying enhancement..."
        )


        enhanced_image = (
            enhance_fundus_image(
                original_image
            )
        )


        enhanced_path = (
            ENHANCED_DIR
            / (
                f"{image_path.stem}"
                f"_enhanced.jpg"
            )
        )


        enhanced_bgr = cv2.cvtColor(
            enhanced_image,
            cv2.COLOR_RGB2BGR
        )


        cv2.imwrite(
            str(enhanced_path),
            enhanced_bgr
        )


        print(
            f"\nEnhanced image saved:"
            f"\n{enhanced_path}"
        )


        # -------------------------------------------------
        # Re-check quality
        # -------------------------------------------------

        (
            enhanced_metrics,
            enhanced_score,
            enhanced_status
        ) = analyze_array_quality(
            enhanced_image
        )


        improvement = (
            enhanced_score
            - original_score
        )


        print(
            "\nENHANCED QUALITY"
        )

        print(
            f"Quality Score: "
            f"{enhanced_score:.2f}/100"
        )

        print(
            f"Quality Status: "
            f"{enhanced_status}"
        )

        print(
            f"Score Change: "
            f"{improvement:+.2f}"
        )


        # -------------------------------------------------
        # Accept enhancement only if improved
        # -------------------------------------------------

        if (
            enhanced_status
            == "GOOD"

            and

            enhanced_score
            > original_score
        ):

            classification_image = (
                enhanced_image
            )

            classification_source = (
                "ENHANCED"
            )

            final_quality_score = (
                enhanced_score
            )

            final_quality_status = (
                enhanced_status
            )


            print(
                "\n✓ Enhancement accepted."
            )

            print(
                "The enhanced image will be "
                "used for classification."
            )


        else:

            print(
                "\n✗ Enhancement did not "
                "produce a sufficiently "
                "good image."
            )

            print(
                "\nRecommendation:"
            )

            print(
                "Recapture fundus image "
                "or request manual review."
            )

            return


    # =====================================================
    # CLASSIFICATION
    # =====================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "REFERABLE DR CLASSIFICATION"
    )

    print(
        "-" * 70
    )


    model = load_model()


    (
        referable_probability,
        decision
    ) = predict_image(
        model,
        classification_image
    )


    print(
        f"\nClassification source:"
        f" {classification_source}"
    )

    print(
        f"Referable probability:"
        f" {referable_probability:.4f}"
    )

    print(
        f"Threshold:"
        f" {THRESHOLD:.2f}"
    )

    print(
        f"Decision:"
        f" {decision}"
    )


    # =====================================================
    # GRAD-CAM
    # =====================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "EXPLAINABILITY"
    )

    print(
        "-" * 70
    )


    (
        heatmap,
        overlay,
        predicted_class,
        gradcam_probability
    ) = generate_gradcam(
        model,
        classification_image
    )


    gradcam_file = (
        OUTPUT_DIR
        / (
            f"{image_path.stem}"
            f"_integrated_gradcam.png"
        )
    )


    save_gradcam(
        image=classification_image,
        heatmap=heatmap,
        overlay=overlay,
        decision=decision,
        probability=referable_probability,
        output_file=gradcam_file
    )


    print(
        "\nGrad-CAM generated successfully."
    )

    print(
        f"Saved:\n{gradcam_file}"
    )


    # =====================================================
    # SCREENING REPORT
    # =====================================================

    report = create_screening_report(

        image_path=image_path,

        original_metrics=(
            original_metrics
        ),

        original_score=(
            original_score
        ),

        original_status=(
            original_status
        ),

        final_quality_score=(
            final_quality_score
        ),

        final_quality_status=(
            final_quality_status
        ),

        classification_source=(
            classification_source
        ),

        referable_probability=(
            referable_probability
        ),

        decision=decision,

        gradcam_file=(
            gradcam_file
        )
    )
    # =========================================================
    # GENERATE SCREENING REPORT
    # =========================================================

    report_file = save_screening_report(
        report,
        image_path
    )

    report_path = str(
        report_file
    )

    print(
        "\n" + "-" * 70
    )

    print(
        "SCREENING REPORT"
    )

    print(
        "-" * 70
    )

    print(
        "\nReport generated successfully."
    )

    print(
        f"Saved:\n{report_file}"
    )

    # =========================================================
    # FINAL RESULT
    # =========================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL SCREENING RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nOriginal quality:"
        f" {original_score:.2f}/100"
    )

    print(
        f"Final quality:"
        f" {final_quality_score:.2f}/100"
    )

    print(
        f"Final quality status:"
        f" {final_quality_status}"
    )

    print(
        f"Classification source:"
        f" {classification_source}"
    )

    print(
        f"Referable probability:"
        f" {referable_probability:.2%}"
    )

    print(
        f"Result:"
        f" {decision}"
    )

    print(
        "Explanation:"
        " Grad-CAM generated ✓"
    )

    print(
        "Screening report:"
        " Generated ✓"
    )

    print(
        "\nNOTE:"
    )

    print(
        "This is an AI-assisted screening "
        "prototype and not a clinical diagnosis."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SCREENING PIPELINE COMPLETE"
    )

    print(
        "=" * 70
    )


    # =========================================================
    # RETURN RESULT TO DJANGO
    # =========================================================

    return {
        "original_quality": float(
            original_score
        ),

        "final_quality": float(
            final_quality_score
        ),

        "quality_status":
            final_quality_status,

        "classification_source":
            classification_source,

        "referable_probability":
            float(
                referable_probability
            ),

        "decision":
            decision,

        "gradcam_generated":
            True,

        "gradcam_path":
            str(
                gradcam_file
            ),
"referable_probability_percent":
    referable_probability * 100,

"report_path":
    report_path,
    }

    # =========================================================
# COMMAND LINE
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python src/inference/predict.py "
            "<image_path>"
        )

        sys.exit(1)

    run_pipeline(
        sys.argv[1]
    )