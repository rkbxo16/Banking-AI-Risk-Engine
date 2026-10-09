# Banking AI Risk Engine

## Credit Risk & Fraud Detection Platform

An end-to-end data science portfolio project exploring how machine learning can support banking risk analysis. The project combines credit-risk modelling, credit-card fraud detection, model explainability, responsible-AI diagnostics and an interactive Streamlit dashboard.

**Status:** In development

## Project Objectives

- Predict potential loan default using supervised machine learning.
- Identify potentially fraudulent card transactions in a highly imbalanced dataset.
- Compare models using metrics suited to risk classification.
- Explain model predictions and investigate fairness considerations.
- Demonstrate model saving, reproducible inference and monitoring.
- Present results through an interactive dashboard.

## Technology Stack

- **Languages:** Python, SQL
- **Data analysis:** Pandas, NumPy
- **Machine learning:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Database:** PostgreSQL
- **Visualisation and application:** Matplotlib, Seaborn, Streamlit
- **Model persistence:** Joblib
- **Development:** Jupyter Notebook, PyCharm, Git

## Project Architecture

Raw Data → Data Validation → Feature Engineering → Model Training → Evaluation → Explainability and Fairness Checks → Risk Scoring → Dashboard and Monitoring

## Fraud Detection

### Dataset

The project uses a publicly available credit-card transaction dataset containing anonymised transaction features, transaction amount and a binary fraud label.

After removing exact duplicate rows, the dataset contains **283,726 transactions**, including **473 fraud cases**. Fraud represents approximately **0.167%** of the cleaned data, making class imbalance an important modelling challenge.

The `V1`–`V28` columns are anonymised transformed features; they do not represent directly interpretable banking attributes.

### Modelling Approach

An XGBoost classifier was trained using a training/test split. A fitted scaler and the trained model are saved for reuse. A decision threshold of **0.20** was selected using out-of-fold threshold analysis.

### Verified Held-Out Test Results

| Metric | Result |
|---|---:|
| Precision | 79.12% |
| Recall | 75.79% |
| F1-score | 77.42% |
| ROC-AUC | 95.60% |
| PR-AUC | 78.63% |

At the 0.20 threshold, the test-set confusion matrix was:

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 56,632 | 19 |
| Actual fraud | 23 | 72 |

These results illustrate the trade-off between catching fraudulent transactions and flagging legitimate transactions for review. The model is a portfolio demonstration, not a production-ready fraud prevention system.

## Credit Risk Modelling

The credit-risk component uses the UCI Statlog German Credit dataset to investigate loan-default risk.

The workflow includes data preparation, categorical encoding, feature scaling, stratified train/test splitting, cross-validation, hyperparameter tuning, model comparison and explainability.

The tuned logistic regression model achieved a held-out test ROC-AUC of **0.810**. This result should be interpreted alongside cross-validation results and the dataset's limitations.

## Explainability and Responsible AI

The project explores:

- Global and local SHAP explanations.
- Group-level performance checks across selected demographic attributes.
- Comparison with a restricted-feature model.
- Data-quality validation and feature-distribution monitoring.

These analyses are diagnostic rather than proof of fairness. Group-level results can be unstable for small samples, and removing sensitive attributes does not necessarily remove proxy effects or disparities.

## Monitoring

Implemented monitoring components include:

- Missing-value, duplicate-row and infinite-value checks.
- Prediction-volume and fraud-flag-rate summaries.
- Risk-band distributions.
- Standardised feature mean-shift analysis.
- Kolmogorov–Smirnov distribution comparisons.

The current drift comparison uses historical training and held-out evaluation data. Production monitoring would require new incoming data, suitable alert thresholds and ongoing performance checks.

## Running the Dashboard

Create and activate a Python virtual environment, then install the project dependencies:

```bash
python -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit dashboard from the project root:

```bash
python -m streamlit run app/streamlit_app.py
```

The saved fraud model, scaler and threshold must be available in the `models/` directory.

## Limitations

- The fraud dataset contains anonymised features and does not reflect all real-world banking transactions.
- Historical test performance does not guarantee performance on future data.
- The dashboard and monitoring components are portfolio prototypes.
- The models have not been approved for real lending or fraud decisions.
- Further work is needed on automated testing, deployment, live-data monitoring and production-grade model governance.

## Future Improvements

- Build an interactive transaction-scoring form.
- Integrate the credit-risk model into the dashboard.
- Add SQL-based data preparation and automated data-quality tests.
- Add automated tests and continuous integration.
- Explore model versioning, containerisation and cloud deployment.
- Improve monitoring with production-oriented drift and performance alerts.

## Disclaimer

This project is for educational and portfolio purposes only. It is not financial advice and must not be used to make real credit or fraud decisions.