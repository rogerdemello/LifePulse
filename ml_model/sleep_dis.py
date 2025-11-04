import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

# ---------------------------
# Step 1: Load the dataset
# ---------------------------
df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset (1).csv")

# Fix hidden spaces in column names (VERY IMPORTANT)
df.columns = df.columns.str.strip()

# ---------------------------
# Step 2: Basic Cleaning
# ---------------------------
df = df.dropna(subset=['Sleep Disorder'])  # Drop rows with missing target

# ---------------------------
# Step 3: Features & Target
# ---------------------------
X = df.drop(columns=['Person ID', 'Sleep Disorder'])
y = df['Sleep Disorder']

# ---------------------------
# Step 4: Encode Categorical Features
# ---------------------------
cat_cols = ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# ---------------------------
# Step 5: Scale Numerical Features
# ---------------------------
num_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
            'Stress Level', 'Heart Rate', 'Daily Steps']

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# ---------------------------
# Step 6: Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------------------
# Step 7: Train Model
# ---------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ---------------------------
# Step 8: Predictions & Evaluation
# ---------------------------
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le_target.classes_))

# ---------------------------
# Step 9: Save Model and Preprocessors
# ---------------------------
output_dir = "saved_sleep_model"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_file = os.path.join(output_dir, f"sleep_rf_model_{timestamp}.pkl")
scaler_file = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
target_encoder_file = os.path.join(output_dir, f"target_encoder_{timestamp}.pkl")
label_encoders_file = os.path.join(output_dir, f"label_encoders_{timestamp}.pkl")

joblib.dump(clf, model_file)
joblib.dump(scaler, scaler_file)
joblib.dump(le_target, target_encoder_file)
joblib.dump(label_encoders, label_encoders_file)

print(f"\nModel saved as: {model_file}")
print(f"Scaler saved as: {scaler_file}")
print(f"Target label encoder saved as: {target_encoder_file}")
print(f"Feature label encoders saved as: {label_encoders_file}")

