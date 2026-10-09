
# --------------------------------------------------
# Load saved fraud detection assets
# --------------------------------------------------



import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.data_quality import check_data_quality
from src.monitoring.model_monitoring import (
    prediction_monitoring,
    risk_band_distribution,
    calculate_feature_drift,
)
from src.monitoring.drift_monitoring import calculate_ks_drift
@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_DIR / "fraud_xgb_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler_fraud.joblib")
    threshold = joblib.load(MODEL_DIR / "fraud_threshold.joblib")
    return model, scaler, threshold

try:
    model, scaler, threshold = load_model_assets()
except Exception as error:
    st.error(f"Could not load the fraud model assets: {error}")
    st.stop()

try:
    feature_names = joblib.load(
        MODEL_DIR / "fraud_feature_names.joblib"
    )
except Exception as error:
    st.error(f"Could not load feature names: {error}")
    st.stop()

@st.cache_data
def load_transaction_data(path):
    data = pd.read_csv(path)

    expected_columns = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
        "Class",
    ]

    missing_columns = [
        column for column in expected_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    return data[expected_columns].copy()


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🏦 Banking AI Risk Engine")
st.subheader("Fraud Detection & Risk Analytics")

st.write(
    "An interactive machine-learning application for "
    "transaction fraud-risk scoring and model monitoring."
)

st.info(
    "Portfolio demonstration using an anonymised credit-card "
    "fraud dataset. Predictions are not financial advice or "
    "production banking decisions."
)

# --------------------------------------------------
# Model overview
# --------------------------------------------------
st.header("Model Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Decision Threshold", f"{threshold:.0%}")

with col2:
    st.metric("Test Recall", "75.79%")

with col3:
    st.metric("Test Precision", "79.12%")

st.caption(
    "Evaluation results from the held-out test set at the "
    "configured threshold. These are historical evaluation "
    "metrics, not live performance measurements."
)

# --------------------------------------------------
# Navigation
# --------------------------------------------------
st.divider()

st.header("Explore the platform")

st.write(
    "Use the sidebar to select a section. The fraud scoring "
    "page will use the saved model and scaler."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Fraud Detection",
        "Model Performance",
        "Monitoring"
    ]
)

if page == "Overview":
    st.subheader("Platform Overview")
    st.markdown(
        """
        - **Fraud Detection:** score a transaction using the saved model.
        - **Model Performance:** review test-set evaluation metrics.
        - **Monitoring:** inspect prediction and data-quality indicators.
        """
    )



elif page == "Fraud Detection":
    st.subheader("Real Transaction Fraud Detection")

    try:
        feature_names = joblib.load(
            MODEL_DIR / "fraud_feature_names.joblib"
        )
        transaction_data = load_transaction_data(str(DATA_PATH))
    except Exception as error:
        st.error(f"Could not load transaction data: {error}")
        st.stop()

    st.write(
        "Select a transaction from the historical dataset and "
        "compare the model's prediction with its known label."
    )

    fraud_only = st.checkbox(
        "Show fraudulent transactions only",
        value=False,
    )

    if fraud_only:
        available_data = transaction_data[
            transaction_data["Class"] == 1
        ]
    else:
        available_data = transaction_data

    st.caption(
        f"{len(available_data):,} transactions available for selection."
    )

    transaction_index = st.number_input(
        "Transaction row index",
        min_value=0,
        max_value=len(available_data) - 1,
        value=0,
        step=1,
    )

    selected_transaction = available_data.iloc[[int(transaction_index)]]

    actual_class = int(selected_transaction["Class"].iloc[0])

    st.markdown("### Selected transaction")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Transaction amount",
            f"{selected_transaction['Amount'].iloc[0]:,.2f}",
        )

    with col2:
        st.metric(
            "Actual dataset label",
            "Fraud" if actual_class == 1 else "Legitimate",
        )

    with st.expander("View transaction features"):
        st.dataframe(
            selected_transaction.drop(columns=["Class"]).T.rename(
                columns={selected_transaction.index[0]: "Value"}
            ),
            use_container_width=True,
        )

    if st.button("Analyse selected transaction", type="primary"):
        # The label must never be passed to the model.
        input_features = selected_transaction.drop(columns=["Class"])
        input_features = input_features[feature_names]

        try:
            scaled_features = scaler.transform(input_features)

            probability = float(
                model.predict_proba(scaled_features)[0, 1]
            )
            predicted_class = int(probability >= threshold)

            st.markdown("### Model assessment")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Estimated fraud probability",
                    f"{probability:.2%}",
                )

            with col2:
                st.metric(
                    "Configured threshold",
                    f"{threshold:.0%}",
                )

            if predicted_class == 1:
                st.error("MODEL DECISION: FLAG FOR REVIEW")
            else:
                st.success("MODEL DECISION: NOT FLAGGED")

            prediction_correct = predicted_class == actual_class

            if prediction_correct:
                st.success("Prediction matches the dataset label.")
            else:
                st.warning("Prediction differs from the dataset label.")

            st.caption(
                "This is a retrospective demonstration using a "
                "labelled historical dataset. The known label is "
                "shown for evaluation only; it is not an input to "
                "the model. This is not a production fraud decision."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


elif page == "Model Performance":
    st.subheader("Fraud Model Performance")

    st.write(
        "Evaluation on the held-out test dataset using the "
        "saved XGBoost model and a 20% decision threshold."
    )

    # Verified results from the current model evaluation
    total_transactions = 56746
    actual_fraud = 95
    true_positives = 72
    false_positives = 19
    false_negatives = 23
    true_negatives = 56632

    precision = true_positives / (
        true_positives + false_positives
    )
    recall = true_positives / (
        true_positives + false_negatives
    )
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (
        true_positives + true_negatives
    ) / total_transactions

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Test transactions", f"{total_transactions:,}")
    col2.metric("Actual fraud cases", f"{actual_fraud:,}")
    col3.metric("Precision", f"{precision:.2%}")
    col4.metric("Recall", f"{recall:.2%}")

    col1, col2, col3 = st.columns(3)

    col1.metric("F1 score", f"{f1_score:.2%}")
    col2.metric("Accuracy", f"{accuracy:.2%}")
    col3.metric("Decision threshold", f"{threshold:.0%}")

    st.divider()

    # Confusion matrix
    st.markdown("### Confusion Matrix")

    confusion_matrix_df = pd.DataFrame(
        [
            [true_negatives, false_positives],
            [false_negatives, true_positives],
        ],
        index=["Actual legitimate", "Actual fraud"],
        columns=["Predicted legitimate", "Predicted fraud"],
    )

    st.dataframe(
        confusion_matrix_df,
        use_container_width=True,
    )

    st.caption(
        "False positives are legitimate transactions flagged for "
        "review. False negatives are fraudulent transactions the "
        "model did not flag."
    )

    # Operational interpretation
    st.markdown("### Operational Interpretation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Fraud cases detected", f"{true_positives}")
    col2.metric("False alarms", f"{false_positives}")
    col3.metric("Missed fraud cases", f"{false_negatives}")

    st.info(
        "At this threshold, the model detected 72 of 95 fraud cases "
        "in the held-out test set and flagged 19 legitimate "
        "transactions. Threshold selection involves a trade-off "
        "between missed fraud and unnecessary reviews."
    )

    st.markdown("### Additional Evaluation Metrics")

    additional_metrics = pd.DataFrame({
        "Metric": ["ROC-AUC", "PR-AUC"],
        "Score": [0.9560, 0.7863],
        "Interpretation": [
            "Ability to rank fraud cases above legitimate cases",
            "Precision-recall performance under severe class imbalance",
        ],
    })

    st.dataframe(additional_metrics, hide_index=True, use_container_width=True)


elif page == "Monitoring":
    st.subheader("Fraud Model Monitoring")

    st.write(
        "Monitor prediction activity, data quality, risk bands "
        "and feature distribution differences."
    )

    try:
        monitoring_data = load_transaction_data(str(DATA_PATH))

        # Separate model inputs from the known historical labels.
        X_monitor = monitoring_data[feature_names].copy()
        y_monitor = monitoring_data["Class"].copy()

        # Score in batches to avoid unnecessarily large transformations.
        probability_parts = []
        batch_size = 10_000

        for start in range(0, len(X_monitor), batch_size):
            batch = X_monitor.iloc[start:start + batch_size]
            scaled_batch = scaler.transform(batch)
            probability_parts.extend(
                model.predict_proba(scaled_batch)[:, 1]
            )

        probabilities = pd.Series(
            probability_parts,
            index=X_monitor.index,
            name="fraud_probability",
        )

        predictions = (probabilities >= threshold).astype(int)

        st.markdown("### Prediction overview")

        report = prediction_monitoring(
            probabilities.to_numpy(),
            predictions.to_numpy(),
            threshold=threshold,
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Transactions scored",
            f"{int(report['number_of_predictions']):,}",
        )
        col2.metric(
            "Average fraud score",
            f"{probabilities.mean():.3%}",
        )
        col3.metric(
            "Transactions flagged",
            f"{int(report['flagged_transactions']):,}",
        )
        col4.metric(
            "Flagged rate",
            f"{report['flagged_rate']:.3%}",
        )

        st.caption(
            f"Configured decision threshold: {threshold:.0%}"
        )

        st.markdown("### Risk band distribution")

        bands = risk_band_distribution(probabilities.to_numpy())

        st.bar_chart(bands)

        st.markdown("### Data quality")

        quality = check_data_quality(
            monitoring_data,
            expected_columns=[
                *feature_names,
                "Class",
            ],
        )

        quality_col1, quality_col2, quality_col3 = st.columns(3)

        quality_col1.metric(
            "Missing values",
            f"{quality['missing_values']:,}",
        )
        quality_col2.metric(
            "Duplicate rows",
            f"{quality['duplicate_rows']:,}",
        )
        quality_col3.metric(
            "Infinite numeric values",
            f"{quality['infinite_values']:,}",
        )

        if quality.get("missing_columns"):
            st.error(
                "Missing columns: "
                + ", ".join(quality["missing_columns"])
            )

        if quality.get("unexpected_columns"):
            st.warning(
                "Unexpected columns: "
                + ", ".join(quality["unexpected_columns"])
            )

        st.markdown("### Feature drift")

        # This is a demonstration comparison between the first
        # and second halves of the historical dataset, not live drift.
        midpoint = len(X_monitor) // 2
        reference_data = X_monitor.iloc[:midpoint]
        current_data = X_monitor.iloc[midpoint:]

        drift_features = [
            "Time", "Amount", "V4", "V10", "V12", "V14"
        ]

        mean_drift = calculate_feature_drift(
            reference_data,
            current_data,
            drift_features,
        )

        ks_drift = calculate_ks_drift(
            reference_data,
            current_data,
            drift_features,
        )

        drift_report = mean_drift.merge(
            ks_drift,
            on="feature",
            how="left",
        )

        drift_report["review_flag"] = (
            drift_report["ks_statistic"] >= 0.10
        )

        st.dataframe(
            drift_report.round(4),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "The KS review flag uses 0.10 as an illustrative "
            "screening threshold, not a universal production rule. "
            "This historical-data comparison does not establish "
            "whether live production data has drifted."
        )

        st.markdown("### Model performance on this dataset")

        actual = y_monitor.to_numpy()
        predicted = predictions.to_numpy()

        true_positives = int(((actual == 1) & (predicted == 1)).sum())
        false_positives = int(((actual == 0) & (predicted == 1)).sum())
        false_negatives = int(((actual == 1) & (predicted == 0)).sum())
        true_negatives = int(((actual == 0) & (predicted == 0)).sum())

        col1, col2, col3 = st.columns(3)

        col1.metric("Fraud detected", f"{true_positives:,}")
        col2.metric("Legitimate transactions flagged", f"{false_positives:,}")
        col3.metric("Fraud missed", f"{false_negatives:,}")

        st.info(
            "These results use the full historical dataset, which "
            "includes the data used during model development. Treat "
            "them as a monitoring demonstration, not an independent "
            "estimate of model performance. Use the held-out test "
            "set for formal evaluation."
        )

    except Exception as error:
        st.error(f"Monitoring could not be completed: {error}")

