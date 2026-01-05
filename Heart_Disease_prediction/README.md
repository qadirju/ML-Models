# Heart Disease Prediction

This folder contains the artifacts for the Heart Disease prediction demo.

Best model: LogisticRegression

Metrics on test set:
- LogisticRegression: accuracy=0.833, AUC=0.950
- DecisionTree: accuracy=0.683, AUC=0.719

Top features (LogisticRegression coefficients):
- ca: 0.959
- thal: 0.739
- cp: 0.508
- sex: 0.484
- oldpeak: 0.445
- trestbps: 0.420
- exang: 0.373
- slope: 0.353
- fbs: 0.331
- restecg: 0.304

Files:
- models/heart_model.joblib
- figures/ (correlation.png, roc.png, cm_*.png, classification reports)
