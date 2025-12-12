import os
import glob
import joblib
import pandas as pd
from flask import Blueprint, render_template, request

sleep_bp = Blueprint('sleep', __name__, url_prefix='/sleep')

# Load encoders for categorical preprocessing
def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return max(files, key=os.path.getctime)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'sleep')

try:
    target_encoder_path = get_latest_file(os.path.join(MODELS_DIR, 'target_encoder_*.pkl'))
    label_encoders_path = get_latest_file(os.path.join(MODELS_DIR, 'label_encoders_*.pkl'))
    
    target_encoder = joblib.load(target_encoder_path)
    label_encoders = joblib.load(label_encoders_path)
    
    # Load model wrapper (local inference, no API)
    from app.utils.onnx_inference import get_model
    model_wrapper = get_model('sleep')
    
    print("✅ Sleep model & encoders loaded successfully!")
except Exception as e:
    print(f"⚠️  Sleep model not found: {e}")
    model_wrapper = None
    target_encoder = None
    label_encoders = None


# ✅ Sleep Disorder Prediction Form
@sleep_bp.route('/', methods=['GET', 'POST'])
def predict_sleep():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        input_data = {}

        # ✅ Field Mapping Between Model & Form Fields
        form_field_mapping = {
            'Gender': 'Gender',
            'Age': 'Age',
            'Occupation': 'Occupation',
            'Sleep Duration': 'SleepDuration',
            'Quality of Sleep': 'QualitySleep',
            'Physical Activity Level': 'PhysicalActivity',
            'Stress Level': 'StressLevel',
            'BMI Category': 'BMICategory',
            'Blood Pressure': 'BloodPressure',
            'Heart Rate': 'HeartRate',
            'Daily Steps': 'DailySteps'
        }

        # ✅ Categorical Columns (encoded)
        cat_cols = ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure']
        for col in cat_cols:
            form_key = form_field_mapping[col]
            value = form_data.get(form_key)
            
            # ✅ Set default for Occupation if not provided
            if value is None:
                if col == 'Occupation':
                    value = 'Nurse'  # Default occupation
                else:
                    return f"Missing value for {col}. Please fill all fields."
            
            try:
                le = label_encoders[col]
                input_data[col] = le.transform([value])[0]
            except ValueError:
                return f"Invalid value '{value}' for {col}. Please enter valid category."

        # ✅ Numerical Columns
        num_cols = ['Age', 'Sleep Duration', 'Quality of Sleep',
                    'Physical Activity Level', 'Stress Level',
                    'Heart Rate', 'Daily Steps']
        for col in num_cols:
            form_key = form_field_mapping[col]
            try:
                input_data[col] = float(form_data.get(form_key))
            except (TypeError, ValueError):
                return f"Invalid value for {col}. Please enter valid numbers."

        # ✅ Arrange in model feature order
        feature_order = ['Gender', 'Age', 'Occupation', 'Sleep Duration',
                         'Quality of Sleep', 'Physical Activity Level',
                         'Stress Level', 'BMI Category', 'Blood Pressure',
                         'Heart Rate', 'Daily Steps']

        # Create input dict for model
        row_dict = {col: input_data[col] for col in feature_order}
        
        # Predict using local model (no API calls)
        y_pred = model_wrapper.predict(row_dict)[0]
        prediction_label = target_encoder.inverse_transform([y_pred])[0]

        return render_template(
            'result_sleep.html',
            prediction=prediction_label,
            input_data=form_data
        )

    return render_template('predict_sleep.html')


