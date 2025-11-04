

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# 🔧 Set save directory (inside app/models/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'models'))
os.makedirs(BASE_DIR, exist_ok=True)

# 1. Load dataset
df = pd.read_csv("data/migraine_dataset_500 (1).csv")
df.columns = df.columns.str.strip()

# 2. Map categorical fields manually
df["Gender"] = df["Gender"].map({'Male': 1, 'Female': 0})
df["Physical Activity"] = df["Physical Activity"].map({
    'None': 0, '1–2 days/week': 1, '3–5 days/week': 2, 'Daily': 3
})
df["Skipped Meals"] = df["Skipped Meals"].map({'Yes': 1, 'No': 0})
df["Menstruating"] = df["Menstruating"].map({
    'No': 0, 'Yes': 1, 'Not applicable': 2
})
df["Migraine"] = df["Migraine"].map({'Yes': 1, 'No': 0})

# 3. Select only features we collect in the form
feature_cols = ['Age', 'Gender', 'Sleep Hours', 'Water Intake', 'Skipped Meals', 
                'Caffeine', 'Stress', 'Screen Time', 'Physical Activity', 'Menstruating']
X = df[feature_cols].copy()
y = df["Migraine"]

# 3.5 Feature Engineering - Add interaction features
X['Sleep_Stress'] = X['Sleep Hours'] * X['Stress']
X['Water_Caffeine'] = X['Water Intake'] * X['Caffeine']
X['Screen_Sleep'] = X['Screen Time'] * X['Sleep Hours']
X['Activity_Stress'] = X['Physical Activity'] * X['Stress']
X['Dehydration_Risk'] = ((X['Caffeine'] > 2) & (X['Water Intake'] < 2)).astype(int)
X['Poor_Sleep_Quality'] = ((X['Sleep Hours'] < 6) | (X['Stress'] > 7)).astype(int)
X['High_Risk_Combo'] = ((X['Stress'] > 6) & (X['Sleep Hours'] < 6) & (X['Caffeine'] > 2)).astype(int)

# 4. Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# 5. No one-hot encoding needed since all features are already numeric
X_encoded = X.copy().fillna(0)

assert X_encoded.dtypes.eq("object").sum() == 0, "Object columns remain!"
assert not X_encoded.isnull().values.any(), "Missing values!"

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 7. SMOTE - balance classes fully
smote = SMOTE(sampling_strategy=1.0, random_state=42)  # Changed from 0.8 to 1.0
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 8. Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# 9. Save Scaler & Columns
pickle.dump(scaler, open(os.path.join(BASE_DIR, "scaler.pkl"), "wb"))
pickle.dump(list(X_encoded.columns), open(os.path.join(BASE_DIR, "columns.pkl"), "wb"))

# 🔍 10. Try XGBoost for better performance
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.5,  # Handle class imbalance
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train_sm)

# 11. Evaluate
y_pred = xgb_model.predict(X_test_scaled)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 12. Save Model & Label Encoder
pickle.dump(xgb_model, open(os.path.join(BASE_DIR, "mindmig_svm_model.pkl"), "wb"))
pickle.dump(le_target, open(os.path.join(BASE_DIR, "label_encoder.pkl"), "wb"))

# 13. Save Confusion Matrix as PNG
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix (SVM)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
confusion_path = os.path.join(BASE_DIR, "svm_confusion_matrix.png")
plt.savefig(confusion_path)
plt.close()

# ✅ Confirmation
print("\n✅ SVM model and all files saved to:")
for file in os.listdir(BASE_DIR):
    print(" -", os.path.join(BASE_DIR, file))
