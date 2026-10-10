
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

def test_saved_model_and_scaler_load():
    model = joblib.load(MODEL_DIR / "fraud_xgb_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler_fraud.joblib")
    threshold = joblib.load(MODEL_DIR / "fraud_threshold.joblib")

    assert model is not None
    assert scaler is not None
    assert 0 < threshold < 1


def test_model_returns_valid_probabilities():
    model = joblib.load(MODEL_DIR / "fraud_xgb_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler_fraud.joblib")
    feature_names = joblib.load(
        MODEL_DIR / "fraud_feature_names.joblib"
    )

    # A synthetic input for a pipeline smoke test, not a real transaction.
    sample = pd.DataFrame(
        [np.zeros(len(feature_names))],
        columns=feature_names,
    )

    scaled_sample = scaler.transform(sample)
    probabilities = model.predict_proba(scaled_sample)

    assert probabilities.shape == (1, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.isclose(probabilities.sum(), 1.0)


def test_threshold_decision_is_consistent():
    threshold = joblib.load(MODEL_DIR / "fraud_threshold.joblib")

    probabilities = np.array([0.05, threshold, 0.85])
    predictions = (probabilities >= threshold).astype(int)

    assert predictions.tolist() == [0, 1, 1]

