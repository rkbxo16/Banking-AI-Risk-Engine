
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def sample_model_and_scaler():
    """Create a small deterministic model for testing."""
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 3.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)

    return model, scaler


def test_model_and_scaler_load(sample_model_and_scaler):
    model, scaler = sample_model_and_scaler

    assert model is not None
    assert scaler is not None
    assert len(model.coef_[0]) == 2


def test_model_returns_valid_probabilities(sample_model_and_scaler):
    model, scaler = sample_model_and_scaler

    sample = np.array([
        [0.0, 0.0],
        [2.0, 3.0],
    ])
    scaled_sample = scaler.transform(sample)
    probabilities = model.predict_proba(scaled_sample)[:, 1]

    assert len(probabilities) == 2
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_threshold_decision_is_consistent():
    probabilities = np.array([0.1, 0.4, 0.8])
    threshold = 0.3

    predictions = (probabilities >= threshold).astype(int)

    assert predictions.tolist() == [0, 1, 1]
