import pandas as pd
import numpy as np


def check_data_quality(df, expected_columns=None):

    results = {}

    # Missing values
    results["missing_values"] = int(
        df.isnull().sum().sum()
    )

    # Duplicate rows
    results["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    # Infinite values
    numeric_df = df.select_dtypes(
        include=np.number
    )

    results["infinite_values"] = int(
        np.isinf(numeric_df).sum().sum()
    )

    # Expected columns
    if expected_columns is not None:

        results["missing_columns"] = list(
            set(expected_columns) - set(df.columns)
        )

        results["unexpected_columns"] = list(
            set(df.columns) - set(expected_columns)
        )

    return results