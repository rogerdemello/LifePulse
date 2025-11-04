"""
Quick test to verify the enhanced migraine model works
Run this after training: python ml_model/test_migraine_model.py
"""
import sys
sys.path.append('.')

import pickle
import numpy as np
import pandas as pd
from app.models import *

print("🧪 Testing Enhanced Migraine Model\n" + "="*50)

# Load model components
try:
    model = pickle.load(open('app/models/mindmig_svm_model.pkl', 'rb'))
    scaler = pickle.load(open('app/models/scaler.pkl', 'rb'))
    columns = pickle.load(open('app/models/columns.pkl', 'rb'))
    label_encoder = pickle.load(open('app/models/label_encoder.pkl', 'rb'))
    print("✅ All model files loaded successfully")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Features: {len(columns)} columns")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Test case 1: High risk patient
print("\n" + "="*50)
print("Test 1: High Risk Patient")
print("="*50)
test_data_1 = {
    'Age': 35, 'Gender': 0, 'Sleep Hours': 4, 'Water Intake': 1,
    'Skipped Meals': 1, 'Caffeine': 4, 'Stress': 9, 'Screen Time': 10,
    'Physical Activity': 0, 'Menstruating': 1
}

df_test = pd.DataFrame([test_data_1])

# Add engineered features (must match training)
df_test['Sleep_Stress'] = df_test['Sleep Hours'] * df_test['Stress']
df_test['Poor_Sleep'] = ((df_test['Sleep Hours'] < 6) | (df_test['Sleep Hours'] > 9)).astype(int)
df_test['Sleep_Quality_Score'] = (df_test['Sleep Hours'] - 7).abs()
df_test['Water_Caffeine_Ratio'] = df_test['Water Intake'] / (df_test['Caffeine'] + 1)
df_test['Dehydration_Risk'] = ((df_test['Caffeine'] > 2) & (df_test['Water Intake'] < 2)).astype(int)
df_test['High_Caffeine'] = (df_test['Caffeine'] > 3).astype(int)
df_test['Screen_Sleep'] = df_test['Screen Time'] * (10 - df_test['Sleep Hours'])
df_test['Activity_Stress'] = df_test['Physical Activity'] * (10 - df_test['Stress'])
df_test['Lifestyle_Risk'] = df_test['Skipped Meals'] + (df_test['Physical Activity'] == 0).astype(int)
df_test['High_Risk_Combo'] = (
    (df_test['Stress'] > 6) & 
    (df_test['Sleep Hours'] < 6) & 
    (df_test['Caffeine'] > 2)
).astype(int)
df_test['Triple_Threat'] = (
    (df_test['Stress'] > 7) & 
    (df_test['Screen Time'] > 6) & 
    (df_test['Sleep Hours'] < 5)
).astype(int)
df_test['Female_Menstruating'] = ((df_test['Gender'] == 0) & (df_test['Menstruating'] == 1)).astype(int)
df_test['Hormonal_Risk'] = df_test['Female_Menstruating'] * (df_test['Stress'] + df_test['Poor_Sleep'])
df_test['Stress_Squared'] = df_test['Stress'] ** 2
df_test['Caffeine_Squared'] = df_test['Caffeine'] ** 2
df_test['Age_Group'] = pd.cut(df_test['Age'], bins=[0, 25, 40, 60, 100], labels=[0, 1, 2, 3]).astype(int)

# Ensure all columns match
df_test = df_test.reindex(columns=columns, fill_value=0)

# Scale and predict
X_scaled = scaler.transform(df_test)
prediction = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None

result = label_encoder.inverse_transform([prediction])[0]
print(f"Prediction: {result}")
if proba is not None:
    print(f"Confidence: No={proba[0]*100:.1f}%, Yes={proba[1]*100:.1f}%")

# Test case 2: Low risk patient
print("\n" + "="*50)
print("Test 2: Low Risk Patient")
print("="*50)
test_data_2 = {
    'Age': 28, 'Gender': 1, 'Sleep Hours': 8, 'Water Intake': 8,
    'Skipped Meals': 0, 'Caffeine': 1, 'Stress': 3, 'Screen Time': 2,
    'Physical Activity': 3, 'Menstruating': 2
}

df_test2 = pd.DataFrame([test_data_2])

# Add all engineered features
df_test2['Sleep_Stress'] = df_test2['Sleep Hours'] * df_test2['Stress']
df_test2['Poor_Sleep'] = ((df_test2['Sleep Hours'] < 6) | (df_test2['Sleep Hours'] > 9)).astype(int)
df_test2['Sleep_Quality_Score'] = (df_test2['Sleep Hours'] - 7).abs()
df_test2['Water_Caffeine_Ratio'] = df_test2['Water Intake'] / (df_test2['Caffeine'] + 1)
df_test2['Dehydration_Risk'] = ((df_test2['Caffeine'] > 2) & (df_test2['Water Intake'] < 2)).astype(int)
df_test2['High_Caffeine'] = (df_test2['Caffeine'] > 3).astype(int)
df_test2['Screen_Sleep'] = df_test2['Screen Time'] * (10 - df_test2['Sleep Hours'])
df_test2['Activity_Stress'] = df_test2['Physical Activity'] * (10 - df_test2['Stress'])
df_test2['Lifestyle_Risk'] = df_test2['Skipped Meals'] + (df_test2['Physical Activity'] == 0).astype(int)
df_test2['High_Risk_Combo'] = (
    (df_test2['Stress'] > 6) & 
    (df_test2['Sleep Hours'] < 6) & 
    (df_test2['Caffeine'] > 2)
).astype(int)
df_test2['Triple_Threat'] = (
    (df_test2['Stress'] > 7) & 
    (df_test2['Screen Time'] > 6) & 
    (df_test2['Sleep Hours'] < 5)
).astype(int)
df_test2['Female_Menstruating'] = ((df_test2['Gender'] == 0) & (df_test2['Menstruating'] == 1)).astype(int)
df_test2['Hormonal_Risk'] = df_test2['Female_Menstruating'] * (df_test2['Stress'] + df_test2['Poor_Sleep'])
df_test2['Stress_Squared'] = df_test2['Stress'] ** 2
df_test2['Caffeine_Squared'] = df_test2['Caffeine'] ** 2
df_test2['Age_Group'] = pd.cut(df_test2['Age'], bins=[0, 25, 40, 60, 100], labels=[0, 1, 2, 3]).astype(int)

df_test2 = df_test2.reindex(columns=columns, fill_value=0)

X_scaled2 = scaler.transform(df_test2)
prediction2 = model.predict(X_scaled2)[0]
proba2 = model.predict_proba(X_scaled2)[0] if hasattr(model, 'predict_proba') else None

result2 = label_encoder.inverse_transform([prediction2])[0]
print(f"Prediction: {result2}")
if proba2 is not None:
    print(f"Confidence: No={proba2[0]*100:.1f}%, Yes={proba2[1]*100:.1f}%")

print("\n" + "="*50)
print("✅ Model testing complete!")
print("="*50)
