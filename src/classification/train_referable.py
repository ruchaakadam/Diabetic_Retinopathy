from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

from torchvision import transforms
from model import create_model


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "referable_train.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "referable_validation.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
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
print("REFERABLE DR BINARY CLASSIFIER")
print("=" * 70)

print(
    f"\nUsing device: {device}"
)


# =========================================================
# Dataset
# =========================================================

class ReferableDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_size=224,
        training=False
    ):

        self.df = pd.read_csv(
            csv_file
        )

        if training:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.RandomRotation(
                    degrees=10
                ),

                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10
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

        else:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406
                    ],
                    std=[
                        0.224,
                        0.225,
                        0.226
                    ]
                )
            ])


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        image = self.transform(
            image
        )

        label = int(
            row["referable"]
        )

        return image, label


# =========================================================
# Create datasets
# =========================================================

train_dataset = ReferableDataset(
    TRAIN_CSV,
    training=True
)

val_dataset = ReferableDataset(
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
    f"\nTraining samples: "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples: "
    f"{len(val_dataset)}"
)


# =========================================================
# Calculate class weights
# =========================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

class_counts = (
    train_df["referable"]
    .value_counts()
    .sort_index()
)

print(
    "\nClass distribution:"
)

print(
    class_counts
)


total = len(
    train_df
)

num_classes = 2

class_weights = []

for label in range(
    num_classes
):

    count = class_counts.get(
        label,
        0
    )

    weight = (
        total
        / (num_classes * count)
    )

    class_weights.append(
        weight
    )


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)


print(
    "\nClass weights:"
)

print(
    class_weights
)


# =========================================================
# Model
# =========================================================

model = create_model(
    num_classes=2
)

model = model.to(
    device
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
    model.classifier.parameters(),
    lr=0.001,
    weight_decay=0.01
)


# =========================================================
# Training
# =========================================================

EPOCHS = 10

best_val_auc = -1.0


for epoch in range(
    EPOCHS
):

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

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

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
    # Training metrics
    # -----------------------------------------------------

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )

    train_f1 = f1_score(
        train_labels,
        train_predictions,
        zero_division=0
    )


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    model.eval()

    val_losses = []

    val_predictions = []

    val_labels = []

    val_probabilities = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

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

            probabilities = torch.softmax(
                outputs,
                dim=1
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

            val_probabilities.extend(
                probabilities[:, 1]
                .cpu()
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
        zero_division=0
    )

    val_auc = roc_auc_score(
        val_labels,
        val_probabilities
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
        f"Train F1: "
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
        f"Val F1: "
        f"{val_f1:.4f}"
    )

    print(
        f"Val ROC-AUC: "
        f"{val_auc:.4f}"
    )


    # -----------------------------------------------------
    # Save best model
    # -----------------------------------------------------

    if val_auc > best_val_auc:

        best_val_auc = val_auc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "✓ Best referable model saved"
        )


# =========================================================
# Final
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    f"Best validation ROC-AUC: "
    f"{best_val_auc:.4f}"
)

print(
    f"Saved model:\n"
    f"{MODEL_PATH}"
)

print(
    "=" * 70
)