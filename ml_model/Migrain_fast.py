"""
FAST VERSION - Quick migraine model training (2-3 minutes)
Use this for testing, then use full Migrain.py for production
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# 🔧 Set save directory (inside app/models/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'models'))
os.makedirs(BASE_DIR, exist_ok=True)

print("🚀 Starting FAST Migraine Model Training...")
print("=" * 60)

# 1. Load dataset
df = pd.read_csv("data/migraine_dataset_500 (1).csv")
df.columns = df.columns.str.strip()
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Map categorical fields
df["Gender"] = df["Gender"].map({'Male': 1, 'Female': 0})
df["Physical Activity"] = df["Physical Activity"].map({
    'None': 0, '1–2 days/week': 1, '3–5 days/week': 2, 'Daily': 3
})
df["Skipped Meals"] = df["Skipped Meals"].map({'Yes': 1, 'No': 0})
df["Menstruating"] = df["Menstruating"].map({
    'No': 0, 'Yes': 1, 'Not applicable': 2
})
df["Migraine"] = df["Migraine"].map({'Yes': 1, 'No': 0})

print(f"📊 Target distribution:\n{df['Migraine'].value_counts()}")

# 3. Select features
feature_cols = ['Age', 'Gender', 'Sleep Hours', 'Water Intake', 'Skipped Meals', 
                'Caffeine', 'Stress', 'Screen Time', 'Physical Activity', 'Menstruating']
X = df[feature_cols].copy()
y = df["Migraine"]

# 3.5 🎯 Feature Engineering
print("\n🔬 Creating features...")
# Sleep-related
X['Sleep_Stress'] = X['Sleep Hours'] * X['Stress']
X['Poor_Sleep'] = ((X['Sleep Hours'] < 6) | (X['Sleep Hours'] > 9)).astype(int)
X['Sleep_Quality_Score'] = (X['Sleep Hours'] - 7).abs()

# Hydration & Caffeine
X['Water_Caffeine_Ratio'] = X['Water Intake'] / (X['Caffeine'] + 1)
X['Dehydration_Risk'] = ((X['Caffeine'] > 2) & (X['Water Intake'] < 2)).astype(int)
X['High_Caffeine'] = (X['Caffeine'] > 3).astype(int)

# Lifestyle
X['Screen_Sleep'] = X['Screen Time'] * (10 - X['Sleep Hours'])
X['Activity_Stress'] = X['Physical Activity'] * (10 - X['Stress'])
X['Lifestyle_Risk'] = X['Skipped Meals'] + (X['Physical Activity'] == 0).astype(int)

# Risk scores
X['High_Risk_Combo'] = (
    (X['Stress'] > 6) & (X['Sleep Hours'] < 6) & (X['Caffeine'] > 2)
).astype(int)
X['Triple_Threat'] = (
    (X['Stress'] > 7) & (X['Screen Time'] > 6) & (X['Sleep Hours'] < 5)
).astype(int)

# Hormonal
X['Female_Menstruating'] = ((X['Gender'] == 0) & (X['Menstruating'] == 1)).astype(int)
X['Hormonal_Risk'] = X['Female_Menstruating'] * (X['Stress'] + X['Poor_Sleep'])

# Polynomial
X['Stress_Squared'] = X['Stress'] ** 2
X['Caffeine_Squared'] = X['Caffeine'] ** 2
X['Age_Group'] = pd.cut(X['Age'], bins=[0, 25, 40, 60, 100], labels=[0, 1, 2, 3]).astype(int)

print(f"✅ Total features: {X.shape[1]}")

# 4. Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# 5. Handle missing
X_encoded = X.copy().fillna(0)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
print(f"\n📊 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# 7. SMOTE
print("⚖️  Applying SMOTE...")
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 8. Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# 9. Save Scaler & Columns
pickle.dump(scaler, open(os.path.join(BASE_DIR, "scaler.pkl"), "wb"))
pickle.dump(list(X_encoded.columns), open(os.path.join(BASE_DIR, "columns.pkl"), "wb"))

# 🎯 10. TRAIN BEST MODELS (no GridSearch for speed)
print("\n🤖 Training models...")
print("=" * 60)

# XGBoost (best params from testing)
print("\n1️⃣  XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.9,
    scale_pos_weight=1.5,
    random_state=42
)
xgb.fit(X_train_scaled, y_train_sm)
xgb_pred = xgb.predict(X_test_scaled)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
print(f"   Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
print(f"   F1 Score: {xgb_f1:.4f}")

# Random Forest (best params)
print("\n2️⃣  Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train_scaled, y_train_sm)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
print(f"   Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
print(f"   F1 Score: {rf_f1:.4f}")

# Select best
best_model = xgb if xgb_f1 >= rf_f1 else rf
best_name = "XGBoost" if xgb_f1 >= rf_f1 else "Random Forest"
best_acc = xgb_acc if xgb_f1 >= rf_f1 else rf_acc
best_f1 = xgb_f1 if xgb_f1 >= rf_f1 else rf_f1

print("\n" + "=" * 60)
print(f"🏆 BEST MODEL: {best_name}")
print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"   F1 Score: {best_f1:.4f}")
print("=" * 60)

# 11. Save Best Model
pickle.dump(best_model, open(os.path.join(BASE_DIR, "mindmig_svm_model.pkl"), "wb"))
pickle.dump(le_target, open(os.path.join(BASE_DIR, "label_encoder.pkl"), "wb"))

# 12. Confusion Matrix
y_pred_best = best_model.predict(X_test_scaled)
print(f"\n📊 Classification Report:\n{classification_report(y_test, y_pred_best)}")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f"Confusion Matrix - {best_name}\nAccuracy: {best_acc*100:.2f}%")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "svm_confusion_matrix.png"))
plt.close()

# 13. Feature Importance
if hasattr(best_model, 'feature_importances_'):
    feature_imp = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 TOP 10 FEATURES:")
    print(feature_imp.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print(f"Model saved: mindmig_svm_model.pkl ({best_name})")
print("=" * 60)
