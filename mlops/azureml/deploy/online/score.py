import os
import logging
import json
import numpy as np
import mlflow


def init():
    """Initialize the scoring environment when the container starts.

    This function is called once when the scoring container is started.
    Use it to load your model and any required resources.
    """
    global model  # pylint: disable=global-statement

    # Get the path to the deployed model
    model_path = os.getenv("AZUREML_MODEL_DIR")

    # TEMPLATE: Load the model using your framework
    # Example with MLflow:
    model = mlflow.pyfunc.load_model(model_path)

    logging.info("Model initialized")


def run(raw_data):
    """Run a model inference on the input data.

    Args:
        raw_data: A JSON string containing the input data.
               Example: '{"data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}'

    Returns:
        A JSON string containing the predictions.
        Example: '{"predictions": [0, 1]}'
    """
    try:
        # Parse the input JSON
        data = json.loads(raw_data)

        # TEMPLATE: Preprocess your data if needed
        # input_data = preprocess(data['data'])

        # TEMPLATE: Make predictions using your model
        predictions = model.predict(data["data"])

        # TEMPLATE: Postprocess predictions if needed
        # results = postprocess(predictions)

        # Return predictions as JSON
        return json.dumps({"predictions": predictions.tolist()})

    except Exception as e:
        error = str(e)
        logging.error(error)
        return json.dumps({"error": error})


# TEMPLATE: Add any helper functions you need
def preprocess(data):
    """Preprocess the input data."""
    # Add your preprocessing logic here
    return data


def postprocess(predictions):
    """Postprocess the model predictions."""
    # Add your postprocessing logic here
    return predictions
