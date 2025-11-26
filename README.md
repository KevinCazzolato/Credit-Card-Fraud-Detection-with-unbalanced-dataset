# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning on a highly imbalanced dataset of European transactions. The dataset contains 284,807 transactions, of which only 492 are fraudulent (~0.17%). Features V1–V28 are anonymized PCA components, with `Time` and `Amount` as original numerical variables.

## Features

- Exploratory Data Analysis (EDA) to uncover patterns in transaction amounts and times.
- Preprocessing including scaling and handling outliers.
- Techniques to handle class imbalance, including class weighting and SMOTE.
- Predictive models: Random Forest and XGBoost with threshold optimization.
- Evaluation using metrics suited for rare events: Precision, Recall, F1-score, ROC-AUC, and Precision-Recall AUC.

## Getting Started

### Prerequisites

Python 3.8+ and the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn
