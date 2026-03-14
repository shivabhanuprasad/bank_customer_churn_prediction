import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

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


#3️⃣ XGBoost Pipeline
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]) # Handling imbalance
    ))
])

xgb_pipeline.fit(X_train, y_train)

y_pred_xgb = xgb_pipeline.predict(X_test)
y_prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

print("🔹 XGBoost Results")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))
print(classification_report(y_test, y_pred_xgb))


#4️⃣ LightGBM Pipeline
lgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    ))
])

lgb_pipeline.fit(X_train, y_train)

y_pred_lgb = lgb_pipeline.predict(X_test)
y_prob_lgb = lgb_pipeline.predict_proba(X_test)[:, 1]

print("🔹 LightGBM Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lgb))
print(classification_report(y_test, y_pred_lgb))


#5️⃣ Save the Best Model automatically based on ROC-AUC
models = {
    "Logistic Regression": (log_reg_pipeline, roc_auc_score(y_test, y_prob_lr)),
    "Random Forest": (rf_pipeline, roc_auc_score(y_test, y_prob_rf)),
    "XGBoost": (xgb_pipeline, roc_auc_score(y_test, y_prob_xgb)),
    "LightGBM": (lgb_pipeline, roc_auc_score(y_test, y_prob_lgb))
}

best_model_name = max(models, key=lambda k: models[k][1])
best_model_pipeline = models[best_model_name][0]

print(f"\n🏆 The Best Model is: {best_model_name} with ROC-AUC = {models[best_model_name][1]:.4f}")

joblib.dump(best_model_pipeline, model_path)
print(f"✅ Best Model saved at: {model_path}")
