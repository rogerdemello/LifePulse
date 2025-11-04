# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime

# === 1. Load CSV ===
data = pd.read_csv('data/Liver_disease_data.csv')  # Replace with actual filename
print("First few rows of data:")
print(data.head())

# === 2. Split features and target ===
X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Feature scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. Train Random Forest model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# === 6. Predictions ===
y_pred = model.predict(X_test_scaled)

# === 7. Evaluation ===
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === 8. Save Model and Scaler ===
output_dir = "saved_liver_model"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(output_dir, f"liver_rf_model_{timestamp}.pkl")
scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
features_path = os.path.join(output_dir, f"features_{timestamp}.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(X.columns.tolist(), features_path)

print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print(f"Feature list saved to: {features_path}")
