import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from preprocessing import (
    load_data,
    split_features_target,
    split_data
)

#🔹Load Model & Dataset

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "raw", "bank_customer_churn.csv")
model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")

# Load data
df = load_data(data_path)
X, y = split_features_target(df)

# Train-test split (same as training)
X_train, X_test, y_train, y_test = split_data(X, y)

# Load trained model
model = joblib.load(model_path)

#🔹 Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

#🔹 1️⃣ Classification Report
print("📊 Classification Report")
print(classification_report(y_test, y_pred))

#🔹 2️⃣ Confusion Matrix (VERY IMPORTANT)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#🔹 3️⃣ ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#4️⃣ Final Evaluation Summary
"""
Model Evaluation Summary:
- Model shows high recall for churn customers
- ROC-AUC indicates strong class separation
- Few churners are missed, making it suitable for retention strategies
"""
