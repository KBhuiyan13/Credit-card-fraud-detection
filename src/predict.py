from pathlib import Path

import joblib
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_model.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "feature_scaler.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "decision_threshold.joblib"


def predict_fraud(transaction_data):
    """
    Predict whether transactions are fraudulent.

    Parameters
    ----------
    transaction_data : pandas.DataFrame
        Transaction features excluding the Class column.

    Returns
    -------
    pandas.DataFrame
        Original data with fraud probability and prediction.
    """

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold_info = joblib.load(THRESHOLD_PATH)

    threshold = threshold_info["threshold"]

    scaled_data = scaler.transform(transaction_data)

    probabilities = model.predict_proba(scaled_data)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    result = transaction_data.copy()

    result["fraud_probability"] = probabilities
    result["fraud_prediction"] = predictions

    return result


if __name__ == "__main__":
    print("Fraud prediction module loaded successfully.")