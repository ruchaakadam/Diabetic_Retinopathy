from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

import matplotlib.pyplot as plt

from dataset import IDRiDDataset
from model import create_model


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "validation.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dr_efficientnet_b0_best.pth"
)


# =========================================================
# Device
# =========================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("=" * 70)
print("DR MODEL EVALUATION")
print("=" * 70)

print(
    f"\nDevice: {device}"
)


# =========================================================
# Dataset
# =========================================================

val_dataset = IDRiDDataset(
    VAL_CSV,
    training=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# =========================================================
# Load model
# =========================================================

model = create_model(
    num_classes=5
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# =========================================================
# Predictions
# =========================================================

all_predictions = []

all_labels = []

all_probabilities = []


with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(
            device
        )

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_probabilities.extend(
            probabilities.cpu()
            .numpy()
        )


# =========================================================
# Metrics
# =========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Macro F1: {macro_f1:.4f}"
)


# =========================================================
# Classification report
# =========================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3, 4],
        target_names=[
            "Grade 0",
            "Grade 1",
            "Grade 2",
            "Grade 3",
            "Grade 4"
        ],
        zero_division=0
    )
)


# =========================================================
# Confusion matrix
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3, 4]
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# =========================================================
# Plot confusion matrix
# =========================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm
)

plt.title(
    "DR Grade Confusion Matrix"
)

plt.xlabel(
    "Predicted Grade"
)

plt.ylabel(
    "Actual Grade"
)

plt.xticks(
    range(5),
    [0, 1, 2, 3, 4]
)

plt.yticks(
    range(5),
    [0, 1, 2, 3, 4]
)

# Add values

for i in range(5):

    for j in range(5):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.show()


# =========================================================
# Referable DR analysis
# =========================================================
#
# Grade 0-1 = non-referable
# Grade 2-4 = referable
# =========================================================

actual_referable = np.array(
    all_labels
) >= 2

predicted_referable = np.array(
    all_predictions
) >= 2


tp = np.sum(
    actual_referable
    & predicted_referable
)

tn = np.sum(
    ~actual_referable
    & ~predicted_referable
)

fp = np.sum(
    ~actual_referable
    & predicted_referable
)

fn = np.sum(
    actual_referable
    & ~predicted_referable
)


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


print("\n" + "=" * 70)
print("REFERABLE DR ANALYSIS")
print("=" * 70)

print(
    "\nReferable DR = Grade 2 or higher"
)

print(
    f"True Positives:  {tp}"
)

print(
    f"True Negatives:  {tn}"
)

print(
    f"False Positives: {fp}"
)

print(
    f"False Negatives: {fn}"
)

print(
    f"\nSensitivity: {sensitivity:.4f}"
)

print(
    f"Specificity: {specificity:.4f}"
)