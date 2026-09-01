from pathlib import Path

from dataset import IDRiDDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.csv"
)

dataset = IDRiDDataset(
    TRAIN_CSV,
    training=True
)

print(
    "Dataset size:",
    len(dataset)
)

image, label = dataset[0]

print(
    "Image shape:",
    image.shape
)

print(
    "Label:",
    label
)