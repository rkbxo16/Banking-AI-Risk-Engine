import pandas as pd
import numpy as np


def prediction_monitoring(
        probabilities,
        predictions,
        threshold=0.20
):
    report = {
        "number_of_predictions": len(predictions),
        "average_fraud_probability": float(
            np.mean(probabilities)
        ),
        "maximum_fraud_probability": float(
            np.max(probabilities)
        ),
        "minimum_fraud_probability": float(
            np.min(probabilities)
        ),
        "flagged_transactions": int(
            np.sum(predictions == 1)
        ),
        "flagged_rate": float(
            np.mean(predictions == 1)
        ),
        "threshold": threshold
    }

    return pd.Series(report)


def risk_band_distribution(probabilities):
    probabilities = np.asarray(probabilities)

    bands = pd.cut(
        probabilities,
        bins=[0, 0.05, 0.20, 0.50, 1.0],
        labels=[
            "Low",
            "Medium",
            "High",
            "Critical"
        ],
        include_lowest=True
    )

    return bands.value_counts().sort_index()

def calculate_feature_drift(
    reference_df,
    current_df,
    features
):
    results = []

    for feature in features:

        reference_mean = reference_df[feature].mean()
        current_mean = current_df[feature].mean()

        reference_std = reference_df[feature].std()

        if reference_std == 0:
            mean_shift = 0
        else:
            mean_shift = (
                (current_mean - reference_mean)
                / reference_std
            )

        results.append({
            "feature": feature,
            "reference_mean": reference_mean,
            "current_mean": current_mean,
            "mean_shift_std": mean_shift
        })

    return pd.DataFrame(results)