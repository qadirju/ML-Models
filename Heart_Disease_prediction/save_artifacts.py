"""Save trained model, figures, and README for Heart Disease Prediction.
This script mirrors the notebook pipeline and writes artifacts into the repository.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import joblib

sns.set(style='whitegrid')

# Data loading
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
print('Downloading dataset...')
df = pd.read_csv(url, header=None, names=cols)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Basic cleaning
print('Cleaning dataset...')
df.replace('?', np.nan, inplace=True)
for c in df.columns:
    try:
        df[c] = pd.to_numeric(df[c])
    except Exception:
        pass

# Drop rows with few missing values
if df.isnull().sum().sum() > 0:
    df = df.dropna()

# Prepare X, y
X = df.drop(columns=['target'])
y = df['target']

# One-hot encode small-cardinality categorical features
cat_cols = [c for c in X.columns if X[c].nunique() <= 10 and X[c].dtype in [np.int64, np.int32, object]]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Train-test split
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Scale for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, solver='liblinear'),
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6)
}

results = {}
for name, model in models.items():
    if name == 'LogisticRegression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    results[name] = {'accuracy': acc, 'cm': cm, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'model': model, 'y_pred': y_pred}

# Prepare folders
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Save correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation matrix')
plt.savefig('figures/correlation.png', bbox_inches='tight')
plt.close()

# Save ROC curves
plt.figure(figsize=(8, 6))
for name, info in results.items():
    plt.plot(info['fpr'], info['tpr'], label=f"{name} (AUC = {info['auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('figures/roc.png', bbox_inches='tight')
plt.close()

# Save confusion matrices and classification reports
for name, info in results.items():
    cm = info['cm']
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['pred0','pred1'], yticklabels=['true0','true1'])
    plt.title(f'Confusion matrix - {name}')
    plt.savefig(f'figures/cm_{name}.png', bbox_inches='tight')
    plt.close()

    report = classification_report(y_test, info['y_pred'])
    with open(f'figures/{name}_classification_report.txt', 'w') as f:
        f.write(report)

# Save best model (by AUC)
best_name = max(results, key=lambda k: results[k]['auc'])
best_model = results[best_name]['model']
joblib.dump({'model': best_model, 'scaler': scaler, 'features': X.columns.tolist()}, 'models/heart_model.joblib')
print(f'Saved best model: {best_name}')

# Write README summary
readme = f"""# Heart Disease Prediction

This folder contains the artifacts for the Heart Disease prediction demo.

Best model: {best_name}

Metrics on test set:
"""
for name, info in results.items():
    readme += f"- {name}: accuracy={info['accuracy']:.3f}, AUC={info['auc']:.3f}\n"

readme += "\nTop features (LogisticRegression coefficients):\n"
coefs = pd.Series(results['LogisticRegression']['model'].coef_.ravel(), index=X.columns)
readme += '\n'.join([f"- {c}: {v:.3f}" for c, v in coefs.abs().sort_values(ascending=False).head(10).items()])
readme += '\n\nFiles:\n- models/heart_model.joblib\n- figures/ (correlation.png, roc.png, cm_*.png, classification reports)\n'

with open('README.md', 'w') as f:
    f.write(readme)

print('Saved README.md and artifacts.')
