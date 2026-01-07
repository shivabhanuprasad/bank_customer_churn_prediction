import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from preprocessing import (
    load_data,
    split_features_target,
    get_column_types,
    build_preprocessor,
    split_data
)

#Load Data & Preprocess

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "raw", "bank_customer_churn.csv")
model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")

# Load data
df = load_data(data_path)
X, y = split_features_target(df)

# Identify columns
cat_cols, num_cols = get_column_types(X)

# Build preprocessor
preprocessor = build_preprocessor(cat_cols, num_cols)

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)


# 1️⃣ Logistic Regression Pipeline
log_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_reg_pipeline.fit(X_train, y_train)

y_pred_lr = log_reg_pipeline.predict(X_test)
y_prob_lr = log_reg_pipeline.predict_proba(X_test)[:, 1]

print("🔹 Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
print(classification_report(y_test, y_pred_lr))

#2️⃣ Random Forest Pipeline
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_pipeline.fit(X_train, y_train)

y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("🔹 Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
print(classification_report(y_test, y_pred_rf))

#3️⃣ Save the Best Model
joblib.dump(rf_pipeline, model_path)
print(f"✅ Model saved at: {model_path}")
