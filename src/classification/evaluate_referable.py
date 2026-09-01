from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score
)

from train_referable import (
    ReferableDataset
)

from model import create_model


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "referable_validation.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dr_referable_best.pth"
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
print("REFERABLE DR MODEL EVALUATION")
print("=" * 70)

print(
    f"\nDevice: {device}"
)


# =========================================================
# Dataset
# =========================================================

dataset = ReferableDataset(
    VAL_CSV,
    training=False
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# =========================================================
# Model
# =========================================================

model = create_model(
    num_classes=2
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

labels = []

probabilities = []


with torch.no_grad():

    for images, batch_labels in loader:

        images = images.to(device)

        outputs = model(
            images
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

        probabilities.extend(
            probs[:, 1]
            .cpu()
            .numpy()
        )

        labels.extend(
            batch_labels.numpy()
        )


labels = np.array(
    labels
)

probabilities = np.array(
    probabilities
)


# =========================================================
# ROC-AUC
# =========================================================

auc = roc_auc_score(
    labels,
    probabilities
)

print(
    f"\nROC-AUC: {auc:.4f}"
)


# =========================================================
# Threshold analysis
# =========================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)


best_threshold = None

best_score = -1


for threshold in np.arange(
    0.10,
    0.91,
    0.05
):

    predictions = (
        probabilities >= threshold
    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()


    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0
    )

    balanced = (
        sensitivity
        + specificity
    ) / 2


    print(
        f"Threshold={threshold:.2f} | "
        f"Sensitivity={sensitivity:.3f} | "
        f"Specificity={specificity:.3f} | "
        f"Balanced={balanced:.3f}"
    )


    # Prefer thresholds satisfying
    # BOTH project requirements.

    if (
        sensitivity >= 0.90
        and
        specificity >= 0.85
    ):

        if balanced > best_score:

            best_score = balanced

            best_threshold = threshold


# =========================================================
# Result
# =========================================================

print("\n" + "=" * 70)

if best_threshold is not None:

    print(
        "TARGET ACHIEVED ON VALIDATION SET"
    )

    print(
        f"Recommended threshold: "
        f"{best_threshold:.2f}"
    )

else:

    print(
        "No threshold simultaneously achieved "
        "90% sensitivity and 85% specificity."
    )

print("=" * 70)