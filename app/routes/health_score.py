import os
import glob
import joblib
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request

health_score_bp = Blueprint('health_score', __name__, url_prefix='/health-score')

# Load model and preprocessors
def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return max(files, key=os.path.getctime)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_regression_model')

model_path = get_latest_file(os.path.join(MODELS_DIR, 'rf_health_model_*.pkl'))
scaler_path = get_latest_file(os.path.join(MODELS_DIR, 'scaler_*.pkl'))
features_path = get_latest_file(os.path.join(MODELS_DIR, 'features_*.pkl'))

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)

print("✅ Health score model loaded successfully!")


def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


@health_score_bp.route('/', methods=['GET', 'POST'])
def predict_health_score():
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            
            # Get base features
            age = int(form_data['Age'])
            bmi = float(form_data['BMI'])
            exercise_freq = int(form_data['ExerciseFrequency'])
            diet_quality = int(form_data['DietQuality'])
            sleep_hours = float(form_data['SleepHours'])
            smoking = int(form_data['SmokingStatus'])
            alcohol = int(form_data['AlcoholConsumption'])
            
            # Create age group
            if age <= 30:
                age_group = 'Young'
            elif age <= 45:
                age_group = 'Mid'
            elif age <= 60:
                age_group = 'Senior'
            else:
                age_group = 'Elder'
            
            # Get BMI category
            bmi_cat = bmi_category(bmi)
            
            # Create feature dict
            input_dict = {
                'Age': age,
                'BMI': bmi,
                'Exercise_Frequency': exercise_freq,
                'Diet_Quality': diet_quality,
                'Sleep_Hours': sleep_hours,
                'Smoking_Status': smoking,
                'Alcohol_Consumption': alcohol,
                'Smoke_Alcohol': smoking * alcohol,
                'Exercise_per_Age': exercise_freq / (age + 1),
                'BMI_squared': bmi ** 2,
                'Sleep_Alcohol': sleep_hours * alcohol,
                'Exercise_Diet': exercise_freq * diet_quality
            }
            
            # Add dummy variables for Age_Group and BMI_Category
            input_dict['Age_Group_Mid'] = 1 if age_group == 'Mid' else 0
            input_dict['Age_Group_Senior'] = 1 if age_group == 'Senior' else 0
            input_dict['Age_Group_Elder'] = 1 if age_group == 'Elder' else 0
            
            input_dict['BMI_Category_Normal'] = 1 if bmi_cat == 'Normal' else 0
            input_dict['BMI_Category_Overweight'] = 1 if bmi_cat == 'Overweight' else 0
            input_dict['BMI_Category_Obese'] = 1 if bmi_cat == 'Obese' else 0
            
            # Create DataFrame with exact feature order
            input_df = pd.DataFrame([input_dict])
            
            # Ensure all features exist
            for feat in feature_names:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            
            input_df = input_df[feature_names]
            
            # Scale numeric features
            numeric_cols = ['Age', 'BMI', 'Exercise_Frequency', 'Diet_Quality', 'Sleep_Hours',
                           'Smoking_Status', 'Alcohol_Consumption', 'Smoke_Alcohol', 'Exercise_per_Age',
                           'BMI_squared', 'Sleep_Alcohol', 'Exercise_Diet']
            
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            
            # Predict
            predicted_score = model.predict(input_df)[0]
            predicted_score = max(0, min(100, predicted_score))  # Clamp between 0-100
            
            # Generate rating with adjusted thresholds for synthetic data
            # Note: Scores are relative to synthetic training data (mean ~85)
            if predicted_score >= 90:
                rating = "Excellent"
                color = "success"
                interpretation = "Outstanding health! You're in the top tier."
            elif predicted_score >= 80:
                rating = "Very Good"
                color = "success"
                interpretation = "Great health profile! Keep up the excellent habits."
            elif predicted_score >= 70:
                rating = "Good"
                color = "primary"
                interpretation = "Solid health foundation with some room for improvement."
            elif predicted_score >= 60:
                rating = "Fair"
                color = "warning"
                interpretation = "Moderate health - focus on making improvements."
            else:
                rating = "Needs Improvement"
                color = "danger"
                interpretation = "Your health needs attention. Consider lifestyle changes."
            
            return render_template('result_health_score.html',
                                 score=round(predicted_score, 1),
                                 rating=rating,
                                 color=color,
                                 interpretation=interpretation,
                                 bmi=bmi,
                                 bmi_cat=bmi_cat,
                                 exercise=exercise_freq,
                                 diet=diet_quality,
                                 sleep=sleep_hours,
                                 smoking=smoking,
                                 alcohol=alcohol)
        
        except Exception as e:
            return f"Error processing form: {str(e)}"
    
    return render_template('predict_health_score.html')
