# Template adapted from Microsoft MLOps examples
# Modify this file according to your project's needs

"""
Registers trained ML model if deploy flag is True.
"""

import argparse
from pathlib import Path
import pickle
import mlflow

import os
import json


def parse_args():
    """Parse input arguments.

    TODO: Customize arguments based on your needs:
    1. Add additional model metadata arguments
    2. Add environment/deployment-specific arguments
    3. Add validation requirements
    """

    parser = argparse.ArgumentParser(description="Model Registration Script")

    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name under which model will be registered",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--evaluation_output",
        type=str,
        required=True,
        help="Path to evaluation results",
    )
    parser.add_argument(
        "--model_info_output_path",
        type=str,
        required=True,
        help="Path to write model info JSON",
    )

    # TODO: Add optional arguments as needed
    # Example:
    # parser.add_argument('--model_version', type=str,
    #                   help='Version tag for the model')
    # parser.add_argument('--environment', type=str,
    #                   help='Deployment environment')

    args = parser.parse_args()
    print(f"Arguments: {args}")

    return args


def main(args):
    """Register the model in MLflow if it meets deployment criteria.

    TODO: Customize this function based on your needs:
    1. Modify model loading logic for your model type
    2. Add custom registration conditions
    3. Add model metadata and tags
    4. Implement versioning strategy
    5. Add model validation steps
    """

    # Read deployment flag
    # TODO: Customize deployment criteria if needed
    with open((Path(args.evaluation_output) / "deploy_flag"), "rb") as infile:
        deploy_flag = int(infile.read())

    mlflow.log_metric("deploy_flag", int(deploy_flag))

    if deploy_flag == 1:
        print(f"Registering model: {args.model_name}")

        # TODO: Modify model loading based on your model type
        # Example: For scikit-learn models
        model = mlflow.sklearn.load_model(args.model_path)

        # TODO: Add custom model validation here if needed
        # Example: Check model attributes, run basic predictions

        # Log model using MLflow
        # TODO: Customize model logging based on your framework
        # Example: For other frameworks, use appropriate mlflow.<framework> module
        mlflow.sklearn.log_model(
            model,
            args.model_name,
            # TODO: Add relevant metadata as needed
            # registered_model_name=args.model_name,
            # metadata={"framework": "scikit-learn", "type": "regression"}
        )

        # Register model in MLflow registry
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{args.model_name}"

        # TODO: Add custom tags or metadata for the registered model
        mlflow_model = mlflow.register_model(
            model_uri,
            args.model_name,
            # tags={"environment": "production", "dataset_version": "1.0"}
        )
        model_version = mlflow_model.version

        # Write model info
        # TODO: Customize model info as needed
        model_info = {
            "id": f"{args.model_name}:{model_version}",
            # Add additional model metadata as needed
            # "framework": "scikit-learn",
            # "type": "regression",
            # "timestamp": datetime.now().isoformat()
        }

        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        with open(output_path, "w") as of:
            json.dump(model_info, fp=of, indent=2)

        print(f"Successfully registered model version: {model_version}")
    else:
        print("Model did not meet registration criteria and will not be registered!")


if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Evaluation output path: {args.evaluation_output}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
