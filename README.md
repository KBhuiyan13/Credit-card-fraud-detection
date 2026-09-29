# End-to-End Credit Card Fraud Detection Pipeline

An end-to-end machine learning project for detecting fraudulent credit card transactions under severe class imbalance.

The project covers exploratory data analysis, preprocessing, baseline modeling, class-imbalance handling, probability-threshold analysis, final evaluation, model serialization, and reusable fraud prediction.

## Project Status

**Completed — Version 1**

## Dataset

This project uses the **Credit Card Fraud Detection** dataset available through Kaggle.

The dataset contains anonymized credit card transactions with a highly imbalanced fraud class.

The raw dataset is intentionally excluded from the Git repository and should be placed locally at:

```text
data/creditcard.csv
```

## Objectives

* Explore the transaction dataset
* Analyze severe class imbalance
* Perform data preprocessing and feature scaling
* Develop a baseline Logistic Regression model
* Investigate class-weighted learning
* Investigate SMOTE-based oversampling
* Compare different imbalance-handling strategies
* Evaluate models using fraud-focused metrics
* Analyze probability classification thresholds
* Select a final operating threshold using validation data
* Evaluate the final model on an untouched test set
* Save the trained model and preprocessing artifacts
* Develop a reusable prediction module

## Methodology

The project uses a stratified train/validation/test workflow.

The following modeling approaches were investigated:

1. **Baseline Logistic Regression**
2. **Class-Weighted Logistic Regression**
3. **SMOTE + Logistic Regression**

The final model is a Logistic Regression model evaluated using a probability threshold of **0.10**.

The final threshold was selected during validation analysis and subsequently evaluated on the held-out test set.

### Evaluation Metrics

Because the dataset is highly imbalanced, the project focuses on:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion Matrix

Accuracy is not treated as the primary metric because it can be misleading for highly imbalanced classification problems.

## Final Test Performance

The final Logistic Regression model was evaluated on the untouched test set using a decision threshold of **0.10**.

| Metric    |  Score |
| --------- | -----: |
| Precision | 0.7115 |
| Recall    | 0.7551 |
| F1-Score  | 0.7327 |
| ROC-AUC   | 0.9537 |
| PR-AUC    | 0.7377 |

### Confusion Matrix

|                       | Predicted Legitimate | Predicted Fraud |
| --------------------- | -------------------: | --------------: |
| **Actual Legitimate** |               56,834 |              30 |
| **Actual Fraud**      |                   24 |              74 |

### Interpretation

At the selected threshold:

* **74** fraudulent transactions were correctly identified.
* **24** fraudulent transactions were classified as legitimate.
* **30** legitimate transactions were incorrectly classified as fraudulent.
* **56,834** legitimate transactions were correctly classified.

The resulting fraud recall was **75.51%**, while precision was **71.15%**.

## Exploratory Data Analysis

The EDA stage examines:

* Dataset structure
* Missing values
* Duplicate records
* Class distribution
* Class imbalance ratio
* Transaction amount distributions
* Transaction-time patterns
* Anonymized feature statistics
* Feature correlations

Generated EDA figures include:

```text
reports/figures/
├── class_distribution.png
├── amount_distribution.png
├── fraud_amount_distribution.png
├── transaction_time_distribution.png
└── correlation_matrix.png
```

## Final Evaluation Outputs

The final model evaluation produces:

```text
reports/
├── final_metrics.csv
└── figures/
    ├── final_roc_curve.png
    ├── final_precision_recall_curve.png
    └── final_confusion_matrix.png
```

## Model Artifacts

The trained model and preprocessing artifacts are saved under:

```text
models/
├── logistic_regression_model.joblib
├── feature_scaler.joblib
└── decision_threshold.joblib
```

These files allow the trained model to be reused without retraining.

## Prediction

A reusable prediction module is provided in:

```text
src/predict.py
```

Example:

```python
import pandas as pd

from src.predict import predict_fraud

transactions = pd.read_csv("path/to/transactions.csv")

results = predict_fraud(transactions)

print(
    results[
        [
            "fraud_probability",
            "fraud_prediction"
        ]
    ]
)
```

The prediction function adds:

* `fraud_probability` — predicted probability of fraud
* `fraud_prediction` — `1` for predicted fraud and `0` for predicted legitimate

Input transactions should contain the same features used during training, excluding the `Class` column.

## Project Structure

```text
.
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── logistic_regression_model.joblib
│   ├── feature_scaler.joblib
│   └── decision_threshold.joblib
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_model_training_and_evaluation.ipynb
│
├── reports/
│   ├── final_metrics.csv
│   └── figures/
│       ├── class_distribution.png
│       ├── amount_distribution.png
│       ├── fraud_amount_distribution.png
│       ├── transaction_time_distribution.png
│       ├── correlation_matrix.png
│       ├── final_roc_curve.png
│       ├── final_precision_recall_curve.png
│       └── final_confusion_matrix.png
│
├── src/
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

## Reproducibility

The project uses a fixed random state:

```python
RANDOM_STATE = 42
```

The final classification threshold is:

```text
0.10
```

The scaler is fitted on the model-training data and subsequently applied to the validation and test data.

## Limitations

* The `V1`–`V28` features are anonymized PCA-derived variables, limiting direct domain interpretation.
* The dataset represents a historical transaction population and may not represent future or production transaction distributions.
* The selected probability threshold is specific to the current dataset and evaluation setup.
* False-positive and false-negative business costs are not explicitly incorporated into threshold optimization.
* The current implementation focuses on Logistic Regression-based approaches.
* The project does not currently include a production API or real-time transaction-processing system.

## Future Improvements

Potential future extensions include:

* XGBoost or other gradient-boosting models
* Hyperparameter optimization
* Probability calibration
* Cost-sensitive threshold optimization
* Temporal validation
* Model drift monitoring
* SHAP-based model explainability
* REST API deployment
* Real-time fraud scoring
* Experiment tracking and model versioning

These are future extensions and are not part of the current implementation.

## Project Workflow

```text
Credit Card Fraud Dataset
          │
          ▼
Exploratory Data Analysis
          │
          ▼
Stratified Train/Test Split
          │
          ▼
Train / Validation Split
          │
          ▼
Feature Scaling
          │
          ├─────────────────────────┐
          ▼                         ▼
Baseline Logistic          Class-Weighted Logistic
Regression                 Regression
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
              SMOTE + Logistic
                 Regression
                       │
                       ▼
             Validation Analysis
                       │
                       ▼
            Threshold Optimization
                       │
                       ▼
             Final Logistic Model
                Threshold = 0.10
                       │
                       ▼
             Held-Out Test Set
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Saved Artifacts      Evaluation Reports
```

## Summary

This project demonstrates a complete machine learning workflow for fraud detection under severe class imbalance.

It progresses from exploratory data analysis through model development, imbalance handling, threshold analysis, final evaluation, model serialization, and reusable prediction.

The current Version 1 implementation intentionally focuses on a clear and reproducible Logistic Regression-based pipeline. More advanced models and explainability techniques can be incorporated in future versions.
