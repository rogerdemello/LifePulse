

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# 🔧 Set save directory (inside app/models/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'models'))
os.makedirs(BASE_DIR, exist_ok=True)

print("🚀 Starting Enhanced Migraine Model Training...")
print("=" * 60)

# 1. Load dataset
df = pd.read_csv("data/migraine_dataset_500 (1).csv")
df.columns = df.columns.str.strip()
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

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

print(f"📊 Target distribution:\n{df['Migraine'].value_counts()}")
print(f"   Class balance: {df['Migraine'].value_counts(normalize=True).to_dict()}")

# 3. Select only features we collect in the form
feature_cols = ['Age', 'Gender', 'Sleep Hours', 'Water Intake', 'Skipped Meals', 
                'Caffeine', 'Stress', 'Screen Time', 'Physical Activity', 'Menstruating']
X = df[feature_cols].copy()
y = df["Migraine"]

# 3.5 🎯 ENHANCED Feature Engineering
print("\n🔬 Creating advanced features...")
# Sleep-related features
X['Sleep_Stress'] = X['Sleep Hours'] * X['Stress']
X['Poor_Sleep'] = ((X['Sleep Hours'] < 6) | (X['Sleep Hours'] > 9)).astype(int)
X['Sleep_Quality_Score'] = (X['Sleep Hours'] - 7).abs()  # Distance from optimal 7h

# Hydration & Caffeine
X['Water_Caffeine_Ratio'] = X['Water Intake'] / (X['Caffeine'] + 1)
X['Dehydration_Risk'] = ((X['Caffeine'] > 2) & (X['Water Intake'] < 2)).astype(int)
X['High_Caffeine'] = (X['Caffeine'] > 3).astype(int)

# Lifestyle risk factors
X['Screen_Sleep'] = X['Screen Time'] * (10 - X['Sleep Hours'])  # More screen + less sleep
X['Activity_Stress'] = X['Physical Activity'] * (10 - X['Stress'])  # Activity reduces stress
X['Lifestyle_Risk'] = X['Skipped Meals'] + (X['Physical Activity'] == 0).astype(int)

# Combined risk scores
X['High_Risk_Combo'] = (
    (X['Stress'] > 6) & 
    (X['Sleep Hours'] < 6) & 
    (X['Caffeine'] > 2)
).astype(int)

X['Triple_Threat'] = (
    (X['Stress'] > 7) & 
    (X['Screen Time'] > 6) & 
    (X['Sleep Hours'] < 5)
).astype(int)

# Menstrual + hormonal factors
X['Female_Menstruating'] = ((X['Gender'] == 0) & (X['Menstruating'] == 1)).astype(int)
X['Hormonal_Risk'] = X['Female_Menstruating'] * (X['Stress'] + X['Poor_Sleep'])

# Polynomial features for key interactions
X['Stress_Squared'] = X['Stress'] ** 2
X['Caffeine_Squared'] = X['Caffeine'] ** 2
X['Age_Group'] = pd.cut(X['Age'], bins=[0, 25, 40, 60, 100], labels=[0, 1, 2, 3]).astype(int)

print(f"✅ Total features: {X.shape[1]} (original: {len(feature_cols)}, engineered: {X.shape[1] - len(feature_cols)})")

# 4. Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# 5. Handle missing values
X_encoded = X.copy().fillna(0)
assert X_encoded.dtypes.eq("object").sum() == 0, "Object columns remain!"
assert not X_encoded.isnull().values.any(), "Missing values!"

# 6. Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
print(f"\n📊 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# 7. SMOTE - balance classes fully
print("⚖️  Applying SMOTE for class balancing...")
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE: {X_train_sm.shape[0]} samples")

# 8. Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# 9. Save Scaler & Columns
pickle.dump(scaler, open(os.path.join(BASE_DIR, "scaler.pkl"), "wb"))
pickle.dump(list(X_encoded.columns), open(os.path.join(BASE_DIR, "columns.pkl"), "wb"))
print("✅ Scaler and columns saved")

# 🎯 10. TRAIN MULTIPLE MODELS WITH HYPERPARAMETER TUNING
print("\n🤖 Training models with GridSearchCV...")
print("=" * 60)

models_to_test = {}

# Model 1: XGBoost with GridSearch
print("\n1️⃣  Training XGBoost...")
xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 1.5, 2]
}
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42),
    xgb_params,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train_scaled, y_train_sm)
models_to_test['XGBoost'] = xgb_grid.best_estimator_
print(f"   Best params: {xgb_grid.best_params_}")
print(f"   Best CV F1: {xgb_grid.best_score_:.4f}")

# Model 2: Random Forest with GridSearch
print("\n2️⃣  Training Random Forest...")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train_scaled, y_train_sm)
models_to_test['Random Forest'] = rf_grid.best_estimator_
print(f"   Best params: {rf_grid.best_params_}")
print(f"   Best CV F1: {rf_grid.best_score_:.4f}")

# Model 3: Gradient Boosting
print("\n3️⃣  Training Gradient Boosting...")
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'subsample': [0.8, 1.0]
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
gb_grid.fit(X_train_scaled, y_train_sm)
models_to_test['Gradient Boosting'] = gb_grid.best_estimator_
print(f"   Best params: {gb_grid.best_params_}")
print(f"   Best CV F1: {gb_grid.best_score_:.4f}")

# Model 4: Voting Ensemble (Combine top 3)
print("\n4️⃣  Creating Voting Ensemble...")
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', models_to_test['XGBoost']),
        ('rf', models_to_test['Random Forest']),
        ('gb', models_to_test['Gradient Boosting'])
    ],
    voting='soft'
)
voting_clf.fit(X_train_scaled, y_train_sm)
models_to_test['Voting Ensemble'] = voting_clf

# 11. EVALUATE ALL MODELS
print("\n" + "=" * 60)
print("📊 MODEL COMPARISON")
print("=" * 60)

results = {}
for name, model in models_to_test.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'accuracy': accuracy, 'f1': f1, 'model': model}
    
    print(f"\n{name}:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   Classification Report:")
    print(classification_report(y_test, y_pred, indent=6))

# 12. SELECT BEST MODEL
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']
best_f1 = results[best_model_name]['f1']

print("\n" + "=" * 60)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"   F1 Score: {best_f1:.4f}")
print("=" * 60)

# 13. Save Best Model & Label Encoder
pickle.dump(best_model, open(os.path.join(BASE_DIR, "mindmig_svm_model.pkl"), "wb"))
pickle.dump(le_target, open(os.path.join(BASE_DIR, "label_encoder.pkl"), "wb"))

# 14. Save Confusion Matrix
y_pred_best = best_model.predict(X_test_scaled)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f"Confusion Matrix - {best_model_name}\nAccuracy: {best_accuracy*100:.2f}%")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
confusion_path = os.path.join(BASE_DIR, "svm_confusion_matrix.png")
plt.savefig(confusion_path)
plt.close()

# 15. Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_imp = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 TOP 10 MOST IMPORTANT FEATURES:")
    print(feature_imp.head(10).to_string(index=False))
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp.head(15)['feature'], feature_imp.head(15)['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Features - {best_model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "feature_importance.png"))
    plt.close()

# ✅ Confirmation
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Model saved: mindmig_svm_model.pkl ({best_model_name})")
print(f"Files saved to: {BASE_DIR}")
print("\nSaved files:")
for file in sorted(os.listdir(BASE_DIR)):
    if file.endswith(('.pkl', '.png')):
        print(f"   - {file}")
print("=" * 60)
