# Template adapted from Microsoft MLOps examples
# Modify this file according to your project's needs

"""
Evaluates trained ML model using test dataset.
Saves predictions, evaluation results and deploy flag.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

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

    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model_name", type=str, help="Name of registered model")
    parser.add_argument("--model_input", type=str, help="Path of input model")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--evaluation_output", type=str, help="Path of eval results")
    parser.add_argument(
        "--runner", type=str, help="Local or Cloud Runner", default="CloudRunner"
    )

    args = parser.parse_args()

    return args


def main(args):
    """Read trained model and test dataset, evaluate model and save result"""

    # Load the test data
    test_data = pd.read_parquet(Path(args.test_data))

    # Split the data into inputs and outputs
    y_test = test_data[TARGET_COL]
    X_test = test_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    # Load the model from input port
    model = mlflow.sklearn.load_model(args.model_input)

    # ---------------- Model Evaluation ---------------- #
    yhat_test, score = model_evaluation(X_test, y_test, model, args.evaluation_output)

    # ----------------- Model Promotion ---------------- #
    if args.runner == "CloudRunner":
        predictions, deploy_flag = model_promotion(
            args.model_name, args.evaluation_output, X_test, y_test, yhat_test, score
        )


def model_evaluation(X_test, y_test, model, evaluation_output):
    """Evaluate model performance on test data.

    TODO: Customize this function based on your specific evaluation needs:
    1. Add/remove metrics based on your problem type (classification/regression)
    2. Modify visualization based on your data and preferences
    3. Add custom evaluation logic specific to your domain
    """

    # Get predictions
    yhat_test = model.predict(X_test)

    # Save predictions
    output_data = X_test.copy()
    output_data["real_label"] = y_test
    output_data["predicted_label"] = yhat_test
    output_data.to_csv((Path(evaluation_output) / "predictions.csv"))

    # TODO: Customize metrics based on your problem type
    # Current metrics are for regression problems
    metrics = {
        "r2": r2_score(y_test, yhat_test),
        "mse": mean_squared_error(y_test, yhat_test),
        "rmse": np.sqrt(mean_squared_error(y_test, yhat_test)),
        "mae": mean_absolute_error(y_test, yhat_test),
    }

    # TODO: Add classification metrics if needed
    # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # metrics.update({
    #     "accuracy": accuracy_score(y_test, yhat_test),
    #     "precision": precision_score(y_test, yhat_test),
    #     "recall": recall_score(y_test, yhat_test),
    #     "f1": f1_score(y_test, yhat_test)
    # })

    # Save metrics to file
    (Path(evaluation_output) / "score.txt").write_text(
        f"Scored with the following model:\n{format(model)}\n\n"
    )
    with open((Path(evaluation_output) / "score.txt"), "a") as outfile:
        for metric_name, value in metrics.items():
            outfile.write(f"{metric_name}: {value:.4f}\n")

    # Log metrics to MLflow
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"test_{metric_name}", value)

    # TODO: Customize visualization based on your needs
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, yhat_test, color="black", alpha=0.5)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="blue",
        linewidth=2,
    )
    plt.xlabel(f"Actual {TARGET_COL}")
    plt.ylabel(f"Predicted {TARGET_COL}")
    plt.title("Model Predictions vs Actual Values")
    plt.tight_layout()
    plt.savefig("predictions.png")
    mlflow.log_artifact("predictions.png")
    plt.close()

    return yhat_test, metrics["r2"]  # TODO: Change return metric if needed


def model_promotion(model_name, evaluation_output, X_test, y_test, yhat_test, score):
    """Compare model performance with previous versions and decide on promotion.

    TODO: Customize this function based on your needs:
    1. Modify the promotion criteria
    2. Add additional comparison metrics
    3. Customize the visualization
    4. Add custom promotion logic
    """

    scores = {}
    predictions = {}

    client = MlflowClient()

    # Compare with previous versions
    for model_run in client.search_model_versions(f"name='{model_name}'"):
        model_version = model_run.version
        mdl = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        predictions[f"{model_name}:{model_version}"] = mdl.predict(X_test)

        # TODO: Customize comparison metric
        scores[f"{model_name}:{model_version}"] = r2_score(
            y_test, predictions[f"{model_name}:{model_version}"]
        )

    # TODO: Customize promotion criteria
    # Current criteria: promote if better than all previous versions
    if scores:
        if score >= max(list(scores.values())):
            deploy_flag = 1
        else:
            deploy_flag = 0
    else:
        deploy_flag = 1
    print(f"Deploy flag: {deploy_flag}")

    # Save deployment decision
    with open((Path(evaluation_output) / "deploy_flag"), "w") as outfile:
        outfile.write(f"{int(deploy_flag)}")

    # Add current model performance
    scores["current_model"] = score
    predictions["current_model"] = yhat_test

    # TODO: Customize visualization
    plt.figure(figsize=(12, 6))
    pd.DataFrame(scores, index=["score"]).plot(kind="bar")
    plt.title("Model Performance Comparison")
    plt.xlabel("Model Version")
    plt.ylabel("Performance Score")
    plt.tight_layout()
    plt.savefig("perf_comparison.png")
    plt.savefig(Path(evaluation_output) / "perf_comparison.png")
    plt.close()

    mlflow.log_metric("deploy_flag", bool(deploy_flag))
    mlflow.log_artifact("perf_comparison.png")

    return predictions, deploy_flag


if __name__ == "__main__":

    mlflow.start_run()

    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_input}",
        f"Test data path: {args.test_data}",
        f"Evaluation output path: {args.evaluation_output}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
