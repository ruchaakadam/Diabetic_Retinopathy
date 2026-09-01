from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from dataset import IDRiDDataset
from model import create_model


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "validation.csv"
)

BASELINE_MODEL = (
    PROJECT_ROOT
    / "models"
    / "dr_efficientnet_b0_best.pth"
)

FINETUNED_MODEL = (
    PROJECT_ROOT
    / "models"
    / "dr_efficientnet_b0_finetuned_best.pth"
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
print("IDRiD EFFICIENTNET FINE-TUNING")
print("=" * 70)

print(
    f"\nUsing device: {device}"
)


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


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


print(
    f"Training samples:   {len(train_dataset)}"
)

print(
    f"Validation samples: {len(val_dataset)}"
)


# =========================================================
# Class weights
# =========================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

class_counts = (
    train_df["grade"]
    .value_counts()
    .sort_index()
)

print("\nClass distribution:")
print(class_counts)


total = len(train_df)

num_classes = 5

class_weights = []

for grade in range(num_classes):

    count = class_counts.get(
        grade,
        0
    )

    if count == 0:

        weight = 0.0

    else:

        weight = total / (
            num_classes * count
        )

    class_weights.append(
        weight
    )


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)


print("\nClass weights:")
print(class_weights)


# =========================================================
# Create model
# =========================================================

model = create_model(
    num_classes=5
)


# =========================================================
# Load our BEST baseline model
# =========================================================

print(
    "\nLoading baseline model..."
)

model.load_state_dict(
    torch.load(
        BASELINE_MODEL,
        map_location="cpu"
    )
)

print(
    "Baseline model loaded."
)


# =========================================================
# Freeze everything first
# =========================================================

for parameter in model.features.parameters():

    parameter.requires_grad = False


# =========================================================
# Unfreeze the LAST TWO EfficientNet blocks
# =========================================================

for parameter in model.features[-2:].parameters():

    parameter.requires_grad = True


# Classifier is trainable
for parameter in model.classifier.parameters():

    parameter.requires_grad = True


# =========================================================
# Move model to device
# =========================================================

model = model.to(device)


# =========================================================
# Show trainable parameters
# =========================================================

trainable = 0
total_parameters = 0

for parameter in model.parameters():

    total_parameters += parameter.numel()

    if parameter.requires_grad:

        trainable += parameter.numel()


print(
    f"\nTrainable parameters: "
    f"{trainable:,}"
)

print(
    f"Total parameters: "
    f"{total_parameters:,}"
)


# =========================================================
# Loss
# =========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.features[-2:].parameters(),
            "lr": 0.00005
        },
        {
            "params": model.classifier.parameters(),
            "lr": 0.0005
        }
    ],
    weight_decay=0.01
)


# =========================================================
# Training settings
# =========================================================

EPOCHS = 15

best_val_f1 = -1.0

patience = 3

epochs_without_improvement = 0


# =========================================================
# Training
# =========================================================

for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.train()

    train_losses = []

    train_predictions = []

    train_labels = []


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        train_losses.append(
            loss.item()
        )


        predictions = (
            outputs.argmax(
                dim=1
            )
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
    # TRAIN METRICS
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
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    val_losses = []

    val_predictions = []

    val_labels = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            val_losses.append(
                loss.item()
            )


            predictions = (
                outputs.argmax(
                    dim=1
                )
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
    # VALIDATION METRICS
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
    # PRINT RESULTS
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
    # SAVE BEST MODEL
    # -----------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            FINETUNED_MODEL
        )

        print(
            "✓ New best fine-tuned model saved"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/"
            f"{patience})"
        )


    # -----------------------------------------------------
    # EARLY STOPPING
    # -----------------------------------------------------

    if epochs_without_improvement >= patience:

        print(
            "\nEarly stopping triggered."
        )

        break


# =========================================================
# Final result
# =========================================================

print("\n" + "=" * 70)

print(
    f"Best fine-tuned validation Macro F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Saved model:\n"
    f"{FINETUNED_MODEL}"
)

print("=" * 70)