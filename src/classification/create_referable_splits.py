from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

TRAIN_FILE = (
    SPLIT_DIR
    / "train.csv"
)

VAL_FILE = (
    SPLIT_DIR
    / "validation.csv"
)


def convert_to_referable(df):

    df = df.copy()

    df["referable"] = (
        df["grade"] >= 2
    ).astype(int)

    return df


def main():

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    val_df = pd.read_csv(
        VAL_FILE
    )

    train_df = convert_to_referable(
        train_df
    )

    val_df = convert_to_referable(
        val_df
    )

    train_output = (
        SPLIT_DIR
        / "referable_train.csv"
    )

    val_output = (
        SPLIT_DIR
        / "referable_validation.csv"
    )

    train_df.to_csv(
        train_output,
        index=False
    )

    val_df.to_csv(
        val_output,
        index=False
    )

    print("=" * 70)
    print("REFERABLE DR DATASET")
    print("=" * 70)

    print("\nTraining:")
    print(
        train_df["referable"]
        .value_counts()
        .sort_index()
    )

    print("\nValidation:")
    print(
        val_df["referable"]
        .value_counts()
        .sort_index()
    )

    print("\nMapping:")
    print("0 = Non-referable (Grade 0–1)")
    print("1 = Referable (Grade 2–4)")

    print("\nSaved:")
    print(train_output)
    print(val_output)


if __name__ == "__main__":
    main()