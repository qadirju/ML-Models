# Heart Disease Prediction ✅

This folder contains a compact reproducible pipeline for predicting the presence of heart disease using the UCI Heart (Cleveland) dataset.

## Overview
- **Objective:** Predict whether a patient has heart disease (binary target: 1 = disease, 0 = no disease).
- **Data source:** UCI Heart Disease (processed Cleveland dataset). The notebook downloads the dataset automatically if no local `heart.csv` is present.
- **Pipeline:** data loading → cleaning → exploratory data analysis → feature preparation → model training (Logistic Regression, Decision Tree) → evaluation → artifact export.

## Results (test split)
- **Best model:** Logistic Regression (selected by AUC)
- **Logistic Regression:** accuracy = **0.833**, AUC = **0.950**
- **Decision Tree:** accuracy = **0.683**, AUC = **0.719**

## Important features
Top features identified (Logistic Regression coefficients): **ca**, **thal**, **cp**, **sex**, **oldpeak**, **trestbps**. These features consistently contributed most to prediction.

## Files included
- `heartDP.ipynb` — Notebook with step-by-step analysis and visualizations.
- `save_artifacts.py` — Script that reproduces training, saves the best model and figures.
- `models/heart_model.joblib` — Saved best model (includes scaler and feature list).
- `figures/` — Exported images and classification reports:
	- `correlation.png`, `roc.png`, `cm_LogisticRegression.png`, `cm_DecisionTree.png`
	- `LogisticRegression_classification_report.txt`, `DecisionTree_classification_report.txt`

## Reproduce results
1. Install dependencies: `pip install -r ../requirements.txt` (or ensure `pandas, scikit-learn, seaborn, matplotlib, joblib` are available).
2. Run the notebook `heartDP.ipynb` or execute:

```powershell
cd Heart_Disease_prediction; python save_artifacts.py
```

Artifacts (models and figures) will be written to `models/` and `figures/` respectively.

## Quick usage example
Load the saved model and predict on new data:

```python
import joblib
obj = joblib.load('models/heart_model.joblib')
model = obj['model']; scaler = obj['scaler']; features = obj['features']
# prepare a DataFrame with the same columns as `features`, then:
# preds = model.predict(scaler.transform(X_new[features]))
```

## Notes & Next steps
- Add cross-validation and hyperparameter tuning for more robust performance estimates (recommended).
- Consider ensemble models (Random Forest or XGBoost) for further improvements.

---
*Created on Jan 05, 2026 — concise reproducible demo for heart disease prediction.*
