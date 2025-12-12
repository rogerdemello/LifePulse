"""
Improved Migraine Model - Simple Fast Training (No Ensemble)
Target: Accuracy > 51%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os
from datetime import datetime
import json

print("=" * 70)
print("IMPROVED MIGRAINE MODEL TRAINING (SIMPLE)")
print("=" * 70)

# Load dataset
df = pd.read_csv("../data/migraine_dataset_500 (1).csv")
df.columns = df.columns.str.strip()

print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Map categorical fields
df["Gender"] = df["Gender"].map({'Male': 1, 'Female': 0})
df["Physical Activity"] = df["Physical Activity"].map({
    'None': 0, '1-2 days/week': 1, '3-5 days/week': 2, 'Daily': 3
})
df["Skipped Meals"] = df["Skipped Meals"].map({'Yes': 1, 'No': 0})
df["Menstruating"] = df["Menstruating"].map({
    'No': 0, 'Yes': 1, 'Not applicable': 2
})
df["Migraine"] = df["Migraine"].map({'Yes': 1, 'No': 0})

print(f"   Target distribution: {df['Migraine'].value_counts().to_dict()}")

# Feature Engineering
print("\nEnhanced Feature Engineering...")

# Select base features
feature_cols = ['Age', 'Gender', 'Sleep Hours', 'Water Intake', 'Skipped Meals', 
                'Caffeine', 'Stress', 'Screen Time', 'Physical Activity', 'Menstruating']
X = df[feature_cols].copy()
y = df["Migraine"]

# Create interaction features
X['Sleep_Stress'] = X['Sleep Hours'] * X['Stress']
X['Water_Caffeine'] = X['Water Intake'] * (1 / (X['Caffeine'] + 1))
X['Activity_Stress_Ratio'] = X['Physical Activity'] / (X['Stress'] + 1)
X['Screen_Sleep_Ratio'] = X['Screen Time'] / (X['Sleep Hours'] + 1)
X['Dehydration_Risk'] = (X['Caffeine'] > 3).astype(int) * (X['Water Intake'] < 4).astype(int)
X['Sleep_Quality'] = ((X['Sleep Hours'] >= 6) & (X['Sleep Hours'] <= 9)).astype(int)
X['High_Risk_Combo'] = ((X['Stress'] > 7) & (X['Water Intake'] < 4)).astype(int)

# Polynomial features for important variables
X['Stress_Squared'] = X['Stress'] ** 2
X['Water_Squared'] = X['Water Intake'] ** 2
X['Sleep_Squared'] = X['Sleep Hours'] ** 2

print(f"   Features after engineering: {X.shape[1]}")

# Handle NaN values
X = X.fillna(X.median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE for class balance
print("\nApplying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"   Original training samples: {len(y_train)}")
print(f"   Balanced training samples: {len(y_train_balanced)}")
print(f"   Class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train_scaled.shape[0]} samples")
print(f"Testing set: {X_test_scaled.shape[0]} samples")

# Train Random Forest
print("\nTraining Random Forest...")
best_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
best_rf.fit(X_train_scaled, y_train_balanced)
print("Random Forest trained successfully")

# Cross-validation
print("\nPerforming 10-Fold Cross-Validation...")
cv_scores = cross_val_score(best_rf, X_train_scaled, y_train_balanced, cv=10, scoring='accuracy')
print(f"10-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Evaluate on test set
y_pred = best_rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Migraine', 'Migraine']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Save models
save_dir = os.path.join('..', 'app', 'models', 'migraine')
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, 'migraine_model_improved.pkl')
scaler_path = os.path.join(save_dir, 'migraine_scaler_improved.pkl')
features_path = os.path.join(save_dir, 'migraine_features_improved.pkl')

joblib.dump(best_rf, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(X.columns.tolist(), features_path)

print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print(f"Features saved to: {features_path}")

# Create model info file
info = {
    'accuracy': accuracy,
    'f1_score': f1,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'model_params': best_rf.get_params(),
    'features': X.columns.tolist(),
    'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

info_path = os.path.join(save_dir, 'model_info_improved.json')
with open(info_path, 'w') as f:
    json.dump(info, f, indent=4)

print(f"Model info saved to: {info_path}")
print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
