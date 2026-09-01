from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "src" / "classification")
)

from model import create_model


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dr_referable_best.pth"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "explainability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("=" * 70)
print("IDRiD GRAD-CAM EXPLAINABILITY")
print("=" * 70)

print(
    f"\nDevice: {DEVICE}"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = create_model(
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)

model.eval()


print(
    "\nModel loaded successfully."
)


# =========================================================
# TRANSFORMATION
# =========================================================

transform = transforms.Compose([

    transforms.ToPILImage(),

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
# GRAD-CAM STORAGE
# =========================================================

activations = None
gradients = None


def forward_hook(
    module,
    input,
    output
):

    global activations

    activations = output


def backward_hook(
    module,
    grad_input,
    grad_output
):

    global gradients

    gradients = grad_output[0]


# =========================================================
# ATTACH HOOKS
# =========================================================

target_layer = model.features[-1]

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# =========================================================
# FIND IMAGE
# =========================================================
def find_image(image_id):

    image_root = (
        PROJECT_ROOT
        / "data"
        / "raw"
    )

    # Search the entire raw dataset
    matches = list(
        image_root.rglob(
            f"{image_id}.jpg"
        )
    )

    if not matches:

        matches = list(
            image_root.rglob(
                f"{image_id}.JPG"
            )
        )

    if not matches:

        matches = list(
            image_root.rglob(
                f"{image_id}*"
            )
        )

    if matches:

        print(
            f"Found image at:\n{matches[0]}"
        )

        return matches[0]

    return None
# =========================================================
# GENERATE GRAD-CAM
# =========================================================
def generate_gradcam(
    model,
    image
):

    global DEVICE

    activations = None
    gradients = None

    # -----------------------------------------------------
    # Hooks
    # -----------------------------------------------------

    def forward_hook(
        module,
        input,
        output
    ):

        nonlocal activations

        activations = output


    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        nonlocal gradients

        gradients = grad_output[0]


    target_layer = model.features[-1]

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )


    try:

        # -------------------------------------------------
        # Prepare image
        # -------------------------------------------------

        tensor = transform(
            Image.fromarray(image)
        )

        tensor = tensor.unsqueeze(
            0
        ).to(DEVICE)

        # THIS IS IMPORTANT
        tensor.requires_grad_(True)


        # -------------------------------------------------
        # Forward
        # -------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )

        output = model(
            tensor
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


        # -------------------------------------------------
        # Backward
        # -------------------------------------------------

        score = output[
            0,
            predicted_class
        ]

        score.backward()


        # -------------------------------------------------
        # Check hooks
        # -------------------------------------------------

        if activations is None:

            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        # -------------------------------------------------
        # Grad-CAM
        # -------------------------------------------------

        acts = activations.detach()

        grads = gradients.detach()


        weights = grads.mean(
            dim=(2, 3),
            keepdim=True
        )


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


        cam = cam.cpu().numpy()


        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        cam -= cam.min()


        if cam.max() > 0:

            cam /= cam.max()


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

        backward_handle.remove()
    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    original = cv2.imread(
        str(image_path)
    )

    if original is None:

        raise ValueError(
            f"Could not load image: "
            f"{image_path}"
        )

    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------------------------------
    # Prepare tensor
    # -----------------------------------------------------

    tensor = transform(
        original
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    # Allow gradients through the input/feature graph.
    tensor.requires_grad_(True)

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    model.zero_grad(
        set_to_none=True
    )

    output = model(
        tensor
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
        probabilities[0, predicted_class]
        .detach()
        .cpu()
        .item()
    )

    # -----------------------------------------------------
    # Backward pass
    # -----------------------------------------------------

    score = output[
        0,
        predicted_class
    ]

    score.backward()

    # -----------------------------------------------------
    # Validate hooks
    # -----------------------------------------------------

    if activations is None:

        raise RuntimeError(
            "Grad-CAM activations were not captured."
        )

    if gradients is None:

        raise RuntimeError(
            "Grad-CAM gradients were not captured."
        )

    # -----------------------------------------------------
    # Extract tensors
    # -----------------------------------------------------

    acts = activations.detach()

    grads = gradients.detach()

    # -----------------------------------------------------
    # Global average pooling of gradients
    # -----------------------------------------------------

    weights = grads.mean(
        dim=(2, 3),
        keepdim=True
    )

    # -----------------------------------------------------
    # Weighted feature maps
    # -----------------------------------------------------

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

    cam = cam.cpu().numpy()

    # -----------------------------------------------------
    # Normalize CAM
    # -----------------------------------------------------

    cam -= cam.min()

    if cam.max() > 0:

        cam /= cam.max()

    # -----------------------------------------------------
    # Resize CAM to original image
    # -----------------------------------------------------

    height, width = (
        original.shape[:2]
    )

    cam = cv2.resize(
        cam,
        (width, height)
    )

    # -----------------------------------------------------
    # Create heatmap
    # -----------------------------------------------------

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------------------------------
    # Overlay
    # -----------------------------------------------------

    overlay = (
        0.55 * original
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
        original,
        heatmap,
        overlay,
        predicted_class,
        probability,
        cam
    )


# =========================================================
# TEST IMAGE
# =========================================================

IMAGE_ID = "IDRiD_118"

image_path = find_image(
    IMAGE_ID
)

if image_path is None:

    raise FileNotFoundError(
        f"Could not find {IMAGE_ID}"
    )


print(
    f"\nAnalyzing: {image_path.name}"
)


(
    original,
    heatmap,
    overlay,
    predicted_class,
    probability,
    cam
) = generate_gradcam(
    image_path
)


# =========================================================
# PREDICTION
# =========================================================

if predicted_class == 1:

    prediction_text = (
        "REFERABLE DR"
    )

else:

    prediction_text = (
        "NON-REFERABLE DR"
    )


print(
    "\nPrediction:"
)

print(
    f"Class: {prediction_text}"
)

print(
    f"Probability: "
    f"{probability:.4f}"
)


# =========================================================
# SAVE VISUALIZATION
# =========================================================

output_file = (
    OUTPUT_DIR
    / f"{IMAGE_ID}_gradcam.png"
)


plt.figure(
    figsize=(15, 5)
)


plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    original
)

plt.title(
    "Original Fundus"
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
    f"{prediction_text}\n"
    f"Probability: {probability:.2f}"
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

plt.show()


print(
    f"\nSaved Grad-CAM:"
)

print(
    output_file
)


print(
    "\n" + "=" * 70
)

print(
    "GRAD-CAM COMPLETE"
)

print(
    "=" * 70
)