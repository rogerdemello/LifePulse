# Import libraries
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

print("🚀 Starting Enhanced Liver Disease Model Training...")
print("=" * 60)

# === 1. Load CSV ===
data = pd.read_csv('data/Liver_disease_data.csv')
print(f"✅ Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"\nColumns: {data.columns.tolist()}")
print(f"\nTarget distribution:\n{data['Diagnosis'].value_counts()}")

# === 2. Feature Engineering ===
print("\n🔬 Creating engineered features...")
X = data.drop(columns=['Diagnosis'])

# Create ratio features (important for liver disease)
if 'SGOT' in X.columns and 'SGPT' in X.columns:
    X['SGOT_SGPT_Ratio'] = X['SGOT'] / (X['SGPT'] + 1)
    X['High_SGOT_SGPT'] = ((X['SGOT'] > 40) & (X['SGPT'] > 40)).astype(int)

if 'Albumin' in X.columns and 'Total_Protein' in X.columns:
    X['Albumin_Globulin_Ratio'] = X['Albumin'] / (X['Total_Protein'] - X['Albumin'] + 0.1)

if 'Direct_Bilirubin' in X.columns and 'Total_Bilirubin' in X.columns:
    X['Indirect_Bilirubin'] = X['Total_Bilirubin'] - X['Direct_Bilirubin']
    X['Bilirubin_Ratio'] = X['Direct_Bilirubin'] / (X['Total_Bilirubin'] + 0.1)

# Age risk groups
if 'Age' in X.columns:
    X['Age_Risk'] = pd.cut(X['Age'], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3]).astype(int)
    X['Age_Squared'] = X['Age'] ** 2

# Enzyme interactions
enzyme_cols = ['Alkaline_Phosphatase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase']
if all(col in X.columns for col in enzyme_cols):
    X['Enzyme_Score'] = X[enzyme_cols].sum(axis=1)
    X['High_Enzyme_Count'] = (X[enzyme_cols] > X[enzyme_cols].median()).sum(axis=1)

print(f"✅ Total features: {X.shape[1]}")

y = data['Diagnosis']

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# === 4. Handle class imbalance with SMOTE ===
print("⚖️  Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE: {X_train_sm.shape[0]} samples")

# === 5. Feature scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# === 6. Train Multiple Models with Hyperparameter Tuning ===
print("\n🤖 Training models...")
print("=" * 60)

models_results = {}

# Model 1: XGBoost
print("\n1️⃣  Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    random_state=42
)
xgb.fit(X_train_scaled, y_train_sm)
xgb_pred = xgb.predict(X_test_scaled)
xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
xgb_auc = roc_auc_score(y_test, xgb_proba) if len(np.unique(y_test)) == 2 else 0
print(f"   Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
print(f"   F1 Score: {xgb_f1:.4f}")
print(f"   AUC-ROC: {xgb_auc:.4f}")
models_results['XGBoost'] = {'model': xgb, 'acc': xgb_acc, 'f1': xgb_f1, 'auc': xgb_auc}

# Model 2: Random Forest
print("\n2️⃣  Training Random Forest...")
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
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
rf_auc = roc_auc_score(y_test, rf_proba) if len(np.unique(y_test)) == 2 else 0
print(f"   Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
print(f"   F1 Score: {rf_f1:.4f}")
print(f"   AUC-ROC: {rf_auc:.4f}")
models_results['Random Forest'] = {'model': rf, 'acc': rf_acc, 'f1': rf_f1, 'auc': rf_auc}

# Model 3: Gradient Boosting
print("\n3️⃣  Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_scaled, y_train_sm)
gb_pred = gb.predict(X_test_scaled)
gb_proba = gb.predict_proba(X_test_scaled)[:, 1]
gb_acc = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred, average='weighted')
gb_auc = roc_auc_score(y_test, gb_proba) if len(np.unique(y_test)) == 2 else 0
print(f"   Accuracy: {gb_acc:.4f} ({gb_acc*100:.2f}%)")
print(f"   F1 Score: {gb_f1:.4f}")
print(f"   AUC-ROC: {gb_auc:.4f}")
models_results['Gradient Boosting'] = {'model': gb, 'acc': gb_acc, 'f1': gb_f1, 'auc': gb_auc}

# Model 4: Voting Ensemble
print("\n4️⃣  Creating Voting Ensemble...")
voting = VotingClassifier(
    estimators=[('xgb', xgb), ('rf', rf), ('gb', gb)],
    voting='soft'
)
voting.fit(X_train_scaled, y_train_sm)
voting_pred = voting.predict(X_test_scaled)
voting_acc = accuracy_score(y_test, voting_pred)
voting_f1 = f1_score(y_test, voting_pred, average='weighted')
print(f"   Accuracy: {voting_acc:.4f} ({voting_acc*100:.2f}%)")
print(f"   F1 Score: {voting_f1:.4f}")
models_results['Voting Ensemble'] = {'model': voting, 'acc': voting_acc, 'f1': voting_f1, 'auc': 0}

# === 7. Select Best Model ===
best_model_name = max(models_results, key=lambda x: models_results[x]['f1'])
best_result = models_results[best_model_name]
best_model = best_result['model']

print("\n" + "=" * 60)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_result['acc']:.4f} ({best_result['acc']*100:.2f}%)")
print(f"   F1 Score: {best_result['f1']:.4f}")
if best_result['auc'] > 0:
    print(f"   AUC-ROC: {best_result['auc']:.4f}")
print("=" * 60)

# === 8. Detailed Evaluation ===
y_pred_best = best_model.predict(X_test_scaled)
print(f"\n📊 Classification Report:\n{classification_report(y_test, y_pred_best)}")
print(f"\n📉 Confusion Matrix:\n{confusion_matrix(y_test, y_pred_best)}")

# === 9. Save Model and Scaler ===
output_dir = os.path.join("app", "models", "liver")
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(output_dir, f"liver_model_{timestamp}.pkl")
scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
features_path = os.path.join(output_dir, f"features_{timestamp}.pkl")

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(X.columns.tolist(), features_path)

print(f"\n✅ Model saved to: {model_path}")
print(f"✅ Scaler saved to: {scaler_path}")
print(f"✅ Feature list saved to: {features_path}")

# === 10. Save Confusion Matrix ===
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title(f"Liver Disease Confusion Matrix - {best_model_name}\nAccuracy: {best_result['acc']*100:.2f}%")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "liver_confusion_matrix.png"))
plt.close()

# === 11. Feature Importance ===
if hasattr(best_model, 'feature_importances_'):
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 TOP 10 MOST IMPORTANT FEATURES:")
    print(feature_imp.head(10).to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp.head(15)['feature'], feature_imp.head(15)['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Features - Liver Disease ({best_model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "liver_feature_importance.png"))
    plt.close()

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
