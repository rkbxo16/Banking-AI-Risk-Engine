
import pandas as pd
import numpy as np

from src.monitoring.data_quality import check_data_quality


def test_clean_data_has_no_quality_issues():
    df = pd.DataFrame({
        "Amount": [10.0, 20.0, 30.0],
        "V1": [0.1, 0.2, 0.3],
    })

    result = check_data_quality(df)

    assert result["missing_values"] == 0
    assert result["duplicate_rows"] == 0
    assert result["infinite_values"] == 0


def test_missing_values_are_detected():
    df = pd.DataFrame({
        "Amount": [10.0, np.nan, 30.0],
        "V1": [0.1, 0.2, 0.3],
    })

    result = check_data_quality(df)

    assert result["missing_values"] == 1


def test_duplicate_rows_are_detected():
    df = pd.DataFrame({
        "Amount": [10.0, 20.0, 20.0],
        "V1": [0.1, 0.2, 0.2],
    })

    result = check_data_quality(df)

    assert result["duplicate_rows"] == 1


def test_infinite_values_are_detected():
    df = pd.DataFrame({
        "Amount": [10.0, np.inf, 30.0],
        "V1": [0.1, 0.2, 0.3],
    })

    result = check_data_quality(df)

    assert result["infinite_values"] == 1


def test_missing_and_unexpected_columns_are_detected():
    df = pd.DataFrame({
        "Amount": [10.0, 20.0],
        "UnexpectedColumn": [1, 2],
    })

    result = check_data_quality(
        df,
        expected_columns=["Amount", "V1"],
    )

    assert result["missing_columns"] == ["V1"]
    assert result["unexpected_columns"] == ["UnexpectedColumn"]
