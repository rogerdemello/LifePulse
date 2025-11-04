import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime

# ---------------------------
# Step 1: Load your dataset
# ---------------------------
data = pd.read_csv("data/heart_disease_health_indicators_BRFSS2015.csv")  # Replace with your full dataset filename

# ---------------------------
# Step 2: Feature Engineering
# ---------------------------

# BMI Categories
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

data['BMI_Category'] = data['BMI'].apply(bmi_category)

# Health Score (Lower is worse)
data['Health_Score'] = data['GenHlth'] + data['PhysHlth'] + data['MentHlth']

# Lifestyle Score (Higher is better)
data['Lifestyle_Score'] = (
    data['PhysActivity'] + data['Fruits'] + data['Veggies'] 
    - data['Smoker'] - data['HvyAlcoholConsump']
)

# ---------------------------
# Step 3: Drop Unnecessary Features
# ---------------------------
drop_cols = ['NoDocbcCost', 'AnyHealthcare']
data.drop(columns=drop_cols, inplace=True)

# ---------------------------
# Step 4: Encode Categorical Features
# ---------------------------
data = pd.get_dummies(data, columns=['BMI_Category'], drop_first=True)

# ---------------------------
# Step 5: Prepare Features & Target
# ---------------------------
X = data.drop("HeartDiseaseorAttack", axis=1)
y = data["HeartDiseaseorAttack"]

# Optional Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Step 6: Split & Train Model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# Step 7: Evaluation
# ---------------------------
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# Step 8: Save Model and Scaler with Timestamp
# ---------------------------
output_dir = "saved_models"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"heart_disease_model_{timestamp}.pkl"
scaler_filename = f"scaler_{timestamp}.pkl"

joblib.dump(model, os.path.join(output_dir, model_filename))
joblib.dump(scaler, os.path.join(output_dir, scaler_filename))

print(f"\nModel saved as: {os.path.join(output_dir, model_filename)}")
print(f"Scaler saved as: {os.path.join(output_dir, scaler_filename)}")

