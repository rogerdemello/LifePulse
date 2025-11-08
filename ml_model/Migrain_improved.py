"""
IMPROVED Migraine Model - Optimized for small datasets
Addresses: overfitting, class imbalance, feature selection
Expected: 70-80% accuracy
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

# Save directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'models'))
os.makedirs(BASE_DIR, exist_ok=True)

print("🚀 Starting IMPROVED Migraine Model Training...")
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
print(f"   Class ratio: {df['Migraine'].value_counts(normalize=True).to_dict()}")

# 3. Base features
feature_cols = ['Age', 'Gender', 'Sleep Hours', 'Water Intake', 'Skipped Meals', 
                'Caffeine', 'Stress', 'Screen Time', 'Physical Activity', 'Menstruating']
X = df[feature_cols].copy()
y = df["Migraine"]

# 4. SELECTIVE Feature Engineering (avoid overfitting)
print("\n🔬 Creating key features (reduced set)...")

# Critical interactions only
X['Sleep_Stress'] = X['Sleep Hours'] * X['Stress']
X['Poor_Sleep'] = ((X['Sleep Hours'] < 6) | (X['Sleep Hours'] > 9)).astype(int)
X['Dehydration_Risk'] = ((X['Caffeine'] > 2) & (X['Water Intake'] < 2)).astype(int)
X['High_Risk_Combo'] = ((X['Stress'] > 6) & (X['Sleep Hours'] < 6) & (X['Caffeine'] > 2)).astype(int)
X['Female_Menstruating'] = ((X['Gender'] == 0) & (X['Menstruating'] == 1)).astype(int)
X['Lifestyle_Risk'] = X['Skipped Meals'] + (X['Physical Activity'] == 0).astype(int)
X['Stress_Squared'] = X['Stress'] ** 2

print(f"✅ Total features: {X.shape[1]}")

# 5. Fill missing values
X = X.fillna(0)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42  # Larger test set for better evaluation
)
print(f"\n📊 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# 7. SMOTE - balanced but not aggressive
print("⚖️  Applying SMOTE...")
smote = SMOTE(sampling_strategy=0.8, random_state=42)  # 80% balance instead of 100%
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE: {X_train_sm.shape[0]} samples")

# 8. Feature Selection - Keep only top K features
print("\n🎯 Selecting top features...")
selector = SelectKBest(f_classif, k=12)  # Keep 12 best features
X_train_selected = selector.fit_transform(X_train_sm, y_train_sm)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()].tolist()
print(f"✅ Selected features: {selected_features}")

# 9. Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# 10. Train models with regularization
print("\n🤖 Training optimized models...")
print("=" * 60)

models = {}

# Model 1: Random Forest (with regularization)
print("\n1️⃣  Random Forest (regularized)...")
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,  # Limited depth to prevent overfitting
    min_samples_split=10,  # Require more samples for split
    min_samples_leaf=5,  # Larger leaf size
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train_sm)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
models['Random Forest'] = {'model': rf, 'acc': rf_acc, 'f1': rf_f1}
print(f"   Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
print(f"   F1 Score: {rf_f1:.4f}")

# Model 2: XGBoost (with regularization)
print("\n2️⃣  XGBoost (regularized)...")
xgb = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,  # Lower learning rate
    max_depth=4,  # Shallow trees
    min_child_weight=3,  # Require more samples
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    scale_pos_weight=1.5,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train_scaled, y_train_sm)
xgb_pred = xgb.predict(X_test_scaled)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
models['XGBoost'] = {'model': xgb, 'acc': xgb_acc, 'f1': xgb_f1}
print(f"   Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
print(f"   F1 Score: {xgb_f1:.4f}")

# Model 3: Gradient Boosting
print("\n3️⃣  Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_scaled, y_train_sm)
gb_pred = gb.predict(X_test_scaled)
gb_acc = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)
models['Gradient Boosting'] = {'model': gb, 'acc': gb_acc, 'f1': gb_f1}
print(f"   Accuracy: {gb_acc:.4f} ({gb_acc*100:.2f}%)")
print(f"   F1 Score: {gb_f1:.4f}")

# 11. Select best model
print("\n" + "="*60)
print("🏆 BEST MODEL SELECTION")
print("="*60)

best_model_name = max(models.keys(), key=lambda k: models[k]['f1'])
best_result = models[best_model_name]

print(f"Winner: {best_model_name}")
print(f"   Accuracy: {best_result['acc']:.4f} ({best_result['acc']*100:.2f}%)")
print(f"   F1 Score: {best_result['f1']:.4f}")

# 12. Cross-validation on best model
print(f"\n🔄 Cross-validation (5-fold) on {best_model_name}...")
cv_scores = cross_val_score(best_result['model'], X_train_scaled, y_train_sm, cv=5, scoring='f1')
print(f"   CV F1 Scores: {cv_scores}")
print(f"   Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# 13. Final predictions and report
y_pred_final = best_result['model'].predict(X_test_scaled)
print(f"\n📊 Final Test Results:")
print(classification_report(y_test, y_pred_final, target_names=['No Migraine', 'Migraine']))

# 14. Save model and artifacts
print("\n💾 Saving model artifacts...")
pickle.dump(best_result['model'], open(os.path.join(BASE_DIR, "mindmig_svm_model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(BASE_DIR, "scaler.pkl"), "wb"))
pickle.dump(selected_features, open(os.path.join(BASE_DIR, "columns.pkl"), "wb"))
pickle.dump(selector, open(os.path.join(BASE_DIR, "feature_selector.pkl"), "wb"))

# Save label encoder (for compatibility)
class DummyEncoder:
    def inverse_transform(self, y):
        return ['No' if i == 0 else 'Yes' for i in y]

dummy_le = DummyEncoder()
pickle.dump(dummy_le, open(os.path.join(BASE_DIR, "label_encoder.pkl"), "wb"))

print(f"✅ Model saved: mindmig_svm_model.pkl")
print(f"✅ Scaler saved: scaler.pkl")
print(f"✅ Features saved: columns.pkl")
print(f"✅ Feature selector saved: feature_selector.pkl")

print(f"\n🎉 Training complete!")
print(f"📈 Final Accuracy: {best_result['acc']*100:.2f}%")
print(f"📈 F1 Score: {best_result['f1']:.4f}")
