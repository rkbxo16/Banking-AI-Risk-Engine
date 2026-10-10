import pandas as pd
from scipy.stats import ks_2samp


def calculate_ks_drift(reference_df, current_df, features):
    results = []

    for feature in features:

        reference = reference_df[feature].dropna()
        current = current_df[feature].dropna()

        ks_statistic, p_value = ks_2samp(
            reference,
            current
        )

        results.append({
            "feature": feature,
            "ks_statistic": ks_statistic,
            "p_value": p_value
        })

    return pd.DataFrame(results)