```python
from pathlib import Path

import joblib
import numpy as np
import pytest

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

MODEL_PATH = MODEL_DIR / "fraud_xgb_model.joblib"
SCALER_PATH = MODEL_DIR / "fraud_scaler.joblib"
THRESHOLD_PATH = MODEL_DIR / "fraud_threshold.joblib"
FEATURES_PATH = MODEL_DIR / "fraud_feature_names.joblib"

MODEL_FILES = [
    MODEL_PATH,
    SCALER_PATH,
    THRESHOLD_PATH,
    FEATURES_PATH,
]


@pytest.mark.skipif(
    not all(path.exists() for path in MODEL_FILES),
    reason="Saved model artefacts are not included in the GitHub repository",
)
def test_saved_model_and_scaler_load():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    assert model is not None
    assert scaler is not None
    assert len(feature_names) > 0


@pytest.mark.skipif(
    not all(path.exists() for path in MODEL_FILES),
    reason="Saved model artefacts are not included in the GitHub repository",
)
def test_model_returns_valid_probabilities():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    sample = np.zeros((2, len(feature_names)))
    scaled_sample = scaler.transform(sample)
    probabilities = model.predict_proba(scaled_sample)[:, 1]

    assert len(probabilities) == 2
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_threshold_decision_is_consistent():
    probabilities = np.array([0.1, 0.4, 0.8])
    threshold = 0.3

    predictions = (probabilities >= threshold).astype(int)

    assert predictions.tolist() == [0, 1, 1]
