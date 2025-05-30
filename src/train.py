# Template adapted from Microsoft MLOps examples
# Modify this file according to your project's needs

"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

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
    """Parse input arguments.

    TODO: Customize arguments based on your needs:
    1. Add model-specific hyperparameters
    2. Add training configuration parameters
    3. Add data preprocessing parameters
    4. Add custom validation parameters
    """

    parser = argparse.ArgumentParser(description="Model Training Script")

    # Data arguments
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training dataset"
    )
    parser.add_argument(
        "--model_output", type=str, required=True, help="Path to save trained model"
    )

    # TODO: Customize model hyperparameters based on your model type
    # Current example is for RandomForestRegressor
    parser.add_argument(
        "--regressor__n_estimators",
        type=int,
        default=500,
        help="Number of trees in the forest",
    )
    parser.add_argument(
        "--regressor__bootstrap",
        type=int,
        default=1,
        help="Whether to use bootstrapping (1=True, 0=False)",
    )
    parser.add_argument(
        "--regressor__max_depth",
        type=int,
        default=10,
        help="Maximum depth of each tree",
    )
    parser.add_argument(
        "--regressor__max_features",
        type=str,
        default="auto",
        help="Number of features to consider at each split",
    )
    parser.add_argument(
        "--regressor__min_samples_leaf",
        type=int,
        default=4,
        help="Minimum samples required at each leaf node",
    )
    parser.add_argument(
        "--regressor__min_samples_split",
        type=int,
        default=5,
        help="Minimum samples required to split a node",
    )

    # TODO: Add custom training arguments as needed
    # Example:
    # parser.add_argument('--learning_rate', type=float, default=0.01,
    #                   help='Learning rate for optimization')
    # parser.add_argument('--batch_size', type=int, default=32,
    #                   help='Training batch size')
    # parser.add_argument('--epochs', type=int, default=100,
    #                   help='Number of training epochs')

    args = parser.parse_args()
    return args


def main(args):
    """Train and save a machine learning model.

    TODO: Customize this function based on your needs:
    1. Add data preprocessing steps
    2. Modify or replace the model
    3. Add custom training logic
    4. Add model validation steps
    5. Customize logging and visualization
    """

    # ------------ Load Data ------------ #
    # TODO: Modify data loading logic for your data format
    train_data = pd.read_parquet(Path(args.train_data))

    # ------------ Preprocessing ------------ #
    # TODO: Add preprocessing steps as needed
    # Example: scaling, encoding, feature engineering

    # Split features and target
    y_train = train_data[TARGET_COL]
    X_train = train_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    # ------------ Model Definition ------------ #
    # TODO: Replace with your chosen model
    # Current example uses RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=args.regressor__n_estimators,
        bootstrap=args.regressor__bootstrap,
        max_depth=args.regressor__max_depth,
        max_features=args.regressor__max_features,
        min_samples_leaf=args.regressor__min_samples_leaf,
        min_samples_split=args.regressor__min_samples_split,
        random_state=0,
    )

    # Log hyperparameters
    # TODO: Update parameters based on your model
    mlflow.log_params(
        {
            "model_type": "RandomForestRegressor",
            "n_estimators": args.regressor__n_estimators,
            "bootstrap": args.regressor__bootstrap,
            "max_depth": args.regressor__max_depth,
            "max_features": args.regressor__max_features,
            "min_samples_leaf": args.regressor__min_samples_leaf,
            "min_samples_split": args.regressor__min_samples_split,
        }
    )

    # ------------ Model Training ------------ #
    # TODO: Add custom training logic if needed
    model.fit(X_train, y_train)

    # ------------ Model Evaluation ------------ #
    # Generate predictions
    yhat_train = model.predict(X_train)

    # Calculate metrics
    # TODO: Customize metrics based on your problem type
    metrics = {
        "train_r2": r2_score(y_train, yhat_train),
        "train_mse": mean_squared_error(y_train, yhat_train),
        "train_rmse": np.sqrt(mean_squared_error(y_train, yhat_train)),
        "train_mae": mean_absolute_error(y_train, yhat_train),
    }

    # Log metrics
    mlflow.log_metrics(metrics)

    # ------------ Visualization ------------ #
    # TODO: Customize visualization based on your needs
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, yhat_train, color="black", alpha=0.5)
    plt.plot(
        [y_train.min(), y_train.max()],
        [y_train.min(), y_train.max()],
        color="blue",
        linewidth=2,
    )
    plt.xlabel(f"Actual {TARGET_COL}")
    plt.ylabel(f"Predicted {TARGET_COL}")
    plt.title("Model Predictions vs Actual Values")
    plt.tight_layout()
    plt.savefig("training_results.png")
    mlflow.log_artifact("training_results.png")
    plt.close()

    # ------------ Save Model ------------ #
    # TODO: Modify model saving based on your framework
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)


if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
        f"n_estimators: {args.regressor__n_estimators}",
        f"bootstrap: {args.regressor__bootstrap}",
        f"max_depth: {args.regressor__max_depth}",
        f"max_features: {args.regressor__max_features}",
        f"min_samples_leaf: {args.regressor__min_samples_leaf}",
        f"min_samples_split: {args.regressor__min_samples_split}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
