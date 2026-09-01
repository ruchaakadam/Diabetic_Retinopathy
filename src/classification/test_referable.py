from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)
from torchvision import transforms

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "src" / "classification")
)

from model import create_model


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dr_referable_best.pth"
)

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "B. Disease Grading"
    / "2. Groundtruths"
    / "b. IDRiD_Disease Grading_Testing Labels.csv"
)

TEST_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "B. Disease Grading"
    / "1. Original Images"
    / "b. Testing Set"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

THRESHOLD = 0.55


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
print("IDRiD OFFICIAL TEST SET - REFERABLE DR EVALUATION")
print("=" * 70)

print(f"\nDevice: {DEVICE}")


# =========================================================
# LOAD TEST LABELS
# =========================================================

df = pd.read_csv(TEST_CSV)

print(
    f"\nOfficial test images: {len(df)}"
)


# =========================================================
# CONVERT GRADE → REFERABLE
# =========================================================

# Grade 0-1 = Non-referable
# Grade 2-4 = Referable

df["referable"] = (
    df["Retinopathy grade"] >= 2
).astype(int)


print("\nTest distribution:")

print(
    df["referable"]
    .value_counts()
    .sort_index()
)

print(
    "\n0 = Non-referable (Grade 0-1)"
)

print(
    "1 = Referable (Grade 2-4)"
)


# =========================================================
# IMAGE TRANSFORMATION
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
    f"\nLoaded model:"
    f"\n{MODEL_PATH}"
)


# =========================================================
# FIND TEST IMAGE
# =========================================================

def find_image(image_name):

    candidates = [
        TEST_IMAGES / image_name,
        TEST_IMAGES / f"{image_name}.jpg",
        TEST_IMAGES / f"{image_name}.JPG",
    ]

    for path in candidates:

        if path.exists():

            return path

    # Recursive fallback
    matches = list(
        TEST_IMAGES.rglob(
            f"{image_name}*"
        )
    )

    if matches:

        return matches[0]

    return None


# =========================================================
# PREDICTION
# =========================================================

y_true = []
y_prob = []
y_pred = []
image_names = []

print("\nRunning predictions...\n")


with torch.no_grad():

    for index, row in df.iterrows():

        image_name = str(
            row["Image name"]
        )

        # CSV names are usually IDRiD_001
        # while files are usually IDRiD_001.jpg

        image_path = find_image(
            image_name
        )

        if image_path is None:

            print(
                f"WARNING: Image not found: "
                f"{image_name}"
            )

            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"WARNING: Could not read: "
                f"{image_path}"
            )

            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        tensor = transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        ).to(DEVICE)

        output = model(
            tensor
        )

        probability = torch.softmax(
            output,
            dim=1
        )[0, 1].item()

        prediction = int(
            probability >= THRESHOLD
        )

        actual = int(
            row["referable"]
        )

        image_names.append(
            image_name
        )

        y_true.append(
            actual
        )

        y_prob.append(
            probability
        )

        y_pred.append(
            prediction
        )

        if (index + 1) % 20 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(df)}"
            )


# =========================================================
# METRICS
# =========================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

accuracy = accuracy_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

balanced = balanced_accuracy_score(
    y_true,
    y_pred
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

cm = confusion_matrix(
    y_true,
    y_pred
)


print(
    f"\nImages evaluated: {len(y_true)}"
)

print(
    f"Threshold: {THRESHOLD:.2f}"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"F1 Score: {f1:.4f}"
)

print(
    f"Balanced Accuracy: {balanced:.4f}"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "CONFUSION MATRIX"
)

print(
    "=" * 70
)

print(
    cm
)


# =========================================================
# SENSITIVITY / SPECIFICITY
# =========================================================

if cm.shape == (2, 2):

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    print(
        f"\nSensitivity: "
        f"{sensitivity:.4f}"
    )

    print(
        f"Specificity: "
        f"{specificity:.4f}"
    )


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 70
)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Non-referable",
            "Referable"
        ],
        digits=4
    )
)


# =========================================================
# SAVE PREDICTIONS
# =========================================================

results = pd.DataFrame({

    "image": image_names,

    "actual_referable": y_true,

    "referable_probability": y_prob,

    "predicted_referable": y_pred

})

output_file = (
    OUTPUT_DIR
    / "referable_test_predictions.csv"
)

results.to_csv(
    output_file,
    index=False
)


print(
    "\nSaved predictions:"
)

print(
    output_file
)

print(
    "\n" + "=" * 70
)

print(
    "OFFICIAL TEST EVALUATION COMPLETE"
)

print(
    "=" * 70
)