# Template adapted from Microsoft MLOps examples
# Modify this file according to your project's needs

"""
Prepares raw data and provides training, validation and test datasets
"""

import argparse

from pathlib import Path
import os
import numpy as np
import pandas as pd

import mlflow

# TODO: Replace with your target column name
TARGET_COL = "${TARGET_COLUMN}"

# TODO: Define your numerical features
# Example format:
# NUMERIC_COLS = [
#     "feature1",
#     "feature2",
#     ...
# ]
NUMERIC_COLS = []

# TODO: Define your categorical nominal features (no inherent order)
# Example format:
# CAT_NOM_COLS = [
#     "category1",
#     "category2",
#     ...
# ]
CAT_NOM_COLS = []

# TODO: Define your categorical ordinal features (have inherent order)
# Example format:
# CAT_ORD_COLS = [
#     "ordinal1",
#     "ordinal2",
#     ...
# ]
CAT_ORD_COLS = []


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--val_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")

    parser.add_argument("--enable_monitoring", type=str, help="enable logging to ADX")
    parser.add_argument(
        "--table_name",
        type=str,
        default="mlmonitoring",
        help="Table name in ADX for logging",
    )

    args = parser.parse_args()

    return args


def log_training_data(df, table_name):
    from obs.collector import Online_Collector

    collector = Online_Collector(table_name)
    collector.batch_collect(df)


def main(args):
    """Read, split, and save datasets.

    TODO: Customize this function based on your needs:
    1. Modify data loading logic for your data format
    2. Add data preprocessing steps
    3. Adjust train/val/test split ratios
    4. Change output format if needed
    """

    # ------------ Reading Data ------------ #
    # TODO: Modify data loading logic for your data format
    data = pd.read_csv((Path(args.raw_data)))

    # Select relevant columns
    # TODO: Add any preprocessing steps here
    data = data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS + [TARGET_COL]]

    # ------------- Split Data ------------- #
    # TODO: Adjust split ratios based on your needs
    split_ratios = {
        "train": 0.7,  # 70% for training
        "val": 0.15,  # 15% for validation
        "test": 0.15,  # 15% for testing
    }

    # Split data into train, val and test datasets
    random_data = np.random.rand(len(data))

    msk_train = random_data < split_ratios["train"]
    msk_val = (random_data >= split_ratios["train"]) & (
        random_data < split_ratios["train"] + split_ratios["val"]
    )
    msk_test = random_data >= (split_ratios["train"] + split_ratios["val"])

    train = data[msk_train]
    val = data[msk_val]
    test = data[msk_test]

    # Log dataset sizes
    mlflow.log_metric("train_size", train.shape[0])
    mlflow.log_metric("val_size", val.shape[0])
    mlflow.log_metric("test_size", test.shape[0])

    # TODO: Modify output format if needed
    train.to_parquet((Path(args.train_data) / "train.parquet"))
    val.to_parquet((Path(args.val_data) / "val.parquet"))
    test.to_parquet((Path(args.test_data) / "test.parquet"))

    # Optional: Enable monitoring
    if (
        args.enable_monitoring.lower() == "true"
        or args.enable_monitoring == "1"
        or args.enable_monitoring.lower() == "yes"
    ):
        log_training_data(data, args.table_name)


if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Val dataset output path: {args.val_data}",
        f"Test dataset path: {args.test_data}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
