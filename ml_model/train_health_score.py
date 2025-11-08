import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime

print("🚀 Training Health Score Model...")

# === 1. Load data ===
data = pd.read_csv('data/synthetic_health_data.csv')
print(f"✅ Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# === 2. Feature Engineering ===
data['Age_Group'] = pd.cut(data['Age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Mid', 'Senior', 'Elder'])

def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

data['BMI_Category'] = data['BMI'].apply(bmi_category)
data['Smoke_Alcohol'] = data['Smoking_Status'] * data['Alcohol_Consumption']
data['Exercise_per_Age'] = data['Exercise_Frequency'] / (data['Age'] + 1)
data['BMI_squared'] = data['BMI'] ** 2
data['Sleep_Alcohol'] = data['Sleep_Hours'] * data['Alcohol_Consumption']
data['Exercise_Diet'] = data['Exercise_Frequency'] * data['Diet_Quality']

# === 3. Encode categorical features ===
data = pd.get_dummies(data, columns=['Age_Group', 'BMI_Category'], drop_first=True)

# === 4. Prepare features and target ===
X = data.drop(columns=['Health_Score'])
y = data['Health_Score']

# === 5. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 6. Scale numeric features ===
numeric_cols = ['Age', 'BMI', 'Exercise_Frequency', 'Diet_Quality', 'Sleep_Hours',
                'Smoking_Status', 'Alcohol_Consumption', 'Smoke_Alcohol', 'Exercise_per_Age',
                'BMI_squared', 'Sleep_Alcohol', 'Exercise_Diet']

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# === 7. Train Random Forest Regressor ===
print("\n🤖 Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# === 8. Predict and evaluate ===
y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Results:")
print(f"   MSE: {mse:.3f}")
print(f"   MAE: {mae:.3f}")
print(f"   R² Score: {r2:.3f} ({r2*100:.1f}%)")

# === 9. Save to app/models/health_score ===
output_dir = "../app/models/health_score"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = os.path.join(output_dir, f"rf_health_model_{timestamp}.pkl")
scaler_filename = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
features_filename = os.path.join(output_dir, f"features_{timestamp}.pkl")

joblib.dump(rf_model, model_filename)
joblib.dump(scaler, scaler_filename)
joblib.dump(X.columns.tolist(), features_filename)

print(f"\n💾 Model saved:")
print(f"   {model_filename}")
print(f"   {scaler_filename}")
print(f"   {features_filename}")
print("\n✅ Training complete!")
