from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score

from dataset import IDRiDDataset
from model import create_model


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "splits" / "validation.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIR
    / "dr_efficientnet_b0_balanced_best.pth"
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
print("BALANCED SAMPLING DR CLASSIFIER")
print("=" * 70)

print(f"\nUsing device: {device}")


# =========================================================
# Dataset
# =========================================================

train_dataset = IDRiDDataset(
    TRAIN_CSV,
    training=True
)

val_dataset = IDRiDDataset(
    VAL_CSV,
    training=False
)


# =========================================================
# WeightedRandomSampler
# =========================================================

train_df = pd.read_csv(TRAIN_CSV)

class_counts = (
    train_df["grade"]
    .value_counts()
    .sort_index()
)

print("\nOriginal training distribution:")
print(class_counts)


# Inverse-frequency class weights
class_weights = {}

for grade in range(5):

    count = class_counts.get(grade, 0)

    if count > 0:
        class_weights[grade] = 1.0 / count
    else:
        class_weights[grade] = 0.0


sample_weights = [
    class_weights[int(grade)]
    for grade in train_df["grade"]
]


sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(
        sample_weights
    ),
    num_samples=len(train_dataset),
    replacement=True
)


# =========================================================
# DataLoaders
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


print(
    f"\nTraining samples per epoch: "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples: "
    f"{len(val_dataset)}"
)


# =========================================================
# Model
# =========================================================

model = create_model(
    num_classes=5
)

model = model.to(device)


# =========================================================
# NORMAL CrossEntropyLoss
#
# IMPORTANT:
# We are NOT using class weights here.
# The sampler handles the imbalance.
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=0.001,
    weight_decay=0.01
)


# =========================================================
# Training
# =========================================================

EPOCHS = 10

best_val_f1 = -1.0


for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    model.train()

    train_losses = []

    train_predictions = []

    train_labels = []


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_losses.append(
            loss.item()
        )

        predictions = outputs.argmax(
            dim=1
        )

        train_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        train_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


    # -----------------------------------------------------
    # Training metrics
    # -----------------------------------------------------

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )

    train_f1 = f1_score(
        train_labels,
        train_predictions,
        average="macro",
        zero_division=0
    )


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    model.eval()

    val_losses = []

    val_predictions = []

    val_labels = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_losses.append(
                loss.item()
            )

            predictions = outputs.argmax(
                dim=1
            )

            val_predictions.extend(
                predictions.cpu()
                .numpy()
            )

            val_labels.extend(
                labels.cpu()
                .numpy()
            )


    # -----------------------------------------------------
    # Validation metrics
    # -----------------------------------------------------

    val_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )

    val_f1 = f1_score(
        val_labels,
        val_predictions,
        average="macro",
        zero_division=0
    )


    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print(
        f"Train Loss: "
        f"{np.mean(train_losses):.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Train Macro F1: "
        f"{train_f1:.4f}"
    )

    print(
        f"Val Loss: "
        f"{np.mean(val_losses):.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Val Macro F1: "
        f"{val_f1:.4f}"
    )


    # -----------------------------------------------------
    # Save best model
    # -----------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "✓ Best balanced model saved"
        )


# =========================================================
# Final
# =========================================================

print("\n" + "=" * 70)

print(
    f"Best validation Macro F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Saved model:\n{MODEL_PATH}"
)

print("=" * 70)