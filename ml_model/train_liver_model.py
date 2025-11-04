# Import libraries
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

print("🚀 Starting Liver Disease Model Training...")
print("=" * 60)

# === 1. Load CSV ===
data = pd.read_csv('data/indian_liver_patient.csv', 
                   names=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                          'Alkaline_Phosphatase', 'Alamine_Aminotransferase', 
                          'Aspartate_Aminotransferase', 'Total_Protein', 
                          'Albumin', 'Albumin_Globulin_Ratio', 'Diagnosis'])

print(f"✅ Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"\nTarget distribution:\n{data['Diagnosis'].value_counts()}")

# Handle missing values
data = data.dropna()
print(f"✅ After removing missing values: {data.shape[0]} rows")

# Encode Gender
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# === 2. Feature Engineering ===
print("\n🔬 Creating engineered features...")
X = data.drop(columns=['Diagnosis'])

# Create ratio features
X['SGOT_SGPT_Ratio'] = X['Aspartate_Aminotransferase'] / (X['Alamine_Aminotransferase'] + 1)
X['High_Enzymes'] = ((X['Aspartate_Aminotransferase'] > 40) & (X['Alamine_Aminotransferase'] > 40)).astype(int)
X['Bilirubin_Ratio'] = X['Direct_Bilirubin'] / (X['Total_Bilirubin'] + 0.1)
X['Indirect_Bilirubin'] = X['Total_Bilirubin'] - X['Direct_Bilirubin']

# Age risk groups
X['Age_Risk'] = pd.cut(X['Age'], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3]).astype(int)
X['Age_Squared'] = X['Age'] ** 2

# Enzyme score
X['Enzyme_Score'] = X['Alkaline_Phosphatase'] + X['Alamine_Aminotransferase'] + X['Aspartate_Aminotransferase']

# Protein deficiency indicator
X['Low_Albumin'] = (X['Albumin'] < 3.5).astype(int)

y = data['Diagnosis']

print(f"✅ Total features after engineering: {X.shape[1]}")
print(f"Feature names: {X.columns.tolist()}")

# === 3. Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n📊 Train set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# === 4. Handle Imbalance with SMOTE ===
print("\n⚖️  Applying SMOTE for class imbalance...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE: {X_train_balanced.shape[0]} samples")
print(f"Class distribution: {pd.Series(y_train_balanced).value_counts()}")

# === 5. Scale Features ===
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# === 6. Train Multiple Models ===
print("\n🤖 Training ensemble models...")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)

# Voting Ensemble
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)],
    voting='soft',
    n_jobs=-1
)

print("Training Voting Ensemble (Random Forest + Gradient Boosting + XGBoost)...")
voting_clf.fit(X_train_scaled, y_train_balanced)

# === 7. Evaluate ===
print("\n📈 Evaluating model...")
y_pred = voting_clf.predict(X_test_scaled)
y_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n{'='*60}")
print(f"🎯 FINAL RESULTS")
print(f"{'='*60}")
print(f"✅ Accuracy: {accuracy*100:.2f}%")
print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ ROC-AUC Score: {roc_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# === 8. Save Model ===
print("\n💾 Saving model...")
os.makedirs('../app/models/liver', exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f'../app/models/liver/liver_model_{timestamp}.pkl'
scaler_path = f'../app/models/liver/scaler_{timestamp}.pkl'
features_path = f'../app/models/liver/features_{timestamp}.pkl'

joblib.dump(voting_clf, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(X.columns.tolist(), features_path)

print(f"✅ Model saved: {model_path}")
print(f"✅ Scaler saved: {scaler_path}")
print(f"✅ Features saved: {features_path}")
print(f"\n🎉 Training complete! Model ready for deployment.")
