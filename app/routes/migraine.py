import os
import pickle
import pandas as pd
from flask import Blueprint, render_template, request

migraine_bp = Blueprint('migraine', __name__, url_prefix='/migraine')

# Load model and preprocessors
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

model = pickle.load(open(os.path.join(MODELS_DIR, 'mindmig_svm_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb'))
columns = pickle.load(open(os.path.join(MODELS_DIR, 'columns.pkl'), 'rb'))
label_encoder = pickle.load(open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'rb'))

print("✅ Migraine model loaded successfully!")


@migraine_bp.route('/', methods=['GET', 'POST'])
def predict_migraine():
    if request.method == 'POST':
        try:
            # Get form data
            form_data = request.form.to_dict()
            
            # Map form inputs to model features
            input_dict = {
                'Age': int(form_data['Age']),
                'Gender': 1 if form_data['Gender'] == 'Male' else 0,
                'Sleep Hours': float(form_data['SleepHours']),
                'Water Intake': float(form_data['WaterIntake']),
                'Skipped Meals': 1 if form_data['SkippedMeals'] == 'Yes' else 0,
                'Caffeine': int(form_data['Caffeine']),
                'Stress': int(form_data['Stress']),
                'Screen Time': float(form_data['ScreenTime']),
                'Physical Activity': int(form_data['PhysicalActivity']),
                'Menstruating': int(form_data['Menstruating'])
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # 🎯 Add ALL engineered features (MUST match training exactly!)
            # Sleep-related features
            input_df['Sleep_Stress'] = input_df['Sleep Hours'] * input_df['Stress']
            input_df['Poor_Sleep'] = ((input_df['Sleep Hours'] < 6) | (input_df['Sleep Hours'] > 9)).astype(int)
            input_df['Sleep_Quality_Score'] = (input_df['Sleep Hours'] - 7).abs()
            
            # Hydration & Caffeine
            input_df['Water_Caffeine_Ratio'] = input_df['Water Intake'] / (input_df['Caffeine'] + 1)
            input_df['Dehydration_Risk'] = ((input_df['Caffeine'] > 2) & (input_df['Water Intake'] < 2)).astype(int)
            input_df['High_Caffeine'] = (input_df['Caffeine'] > 3).astype(int)
            
            # Lifestyle risk factors
            input_df['Screen_Sleep'] = input_df['Screen Time'] * (10 - input_df['Sleep Hours'])
            input_df['Activity_Stress'] = input_df['Physical Activity'] * (10 - input_df['Stress'])
            input_df['Lifestyle_Risk'] = input_df['Skipped Meals'] + (input_df['Physical Activity'] == 0).astype(int)
            
            # Combined risk scores
            input_df['High_Risk_Combo'] = (
                (input_df['Stress'] > 6) & 
                (input_df['Sleep Hours'] < 6) & 
                (input_df['Caffeine'] > 2)
            ).astype(int)
            
            input_df['Triple_Threat'] = (
                (input_df['Stress'] > 7) & 
                (input_df['Screen Time'] > 6) & 
                (input_df['Sleep Hours'] < 5)
            ).astype(int)
            
            # Menstrual + hormonal factors
            input_df['Female_Menstruating'] = ((input_df['Gender'] == 0) & (input_df['Menstruating'] == 1)).astype(int)
            input_df['Hormonal_Risk'] = input_df['Female_Menstruating'] * (input_df['Stress'] + input_df['Poor_Sleep'])
            
            # Polynomial features
            input_df['Stress_Squared'] = input_df['Stress'] ** 2
            input_df['Caffeine_Squared'] = input_df['Caffeine'] ** 2
            input_df['Age_Group'] = pd.cut(input_df['Age'], bins=[0, 25, 40, 60, 100], labels=[0, 1, 2, 3]).astype(int)
            
            # Ensure column order matches training
            input_df = input_df.reindex(columns=columns, fill_value=0)
            
            # Scale
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Decode prediction
            result = 'Migraine Risk' if prediction == 1 else 'No Migraine Risk'
            confidence = max(prediction_proba) * 100
            
            return render_template('result_migraine.html', 
                                 prediction=result,
                                 confidence=round(confidence, 1))
        
        except Exception as e:
            return f"Error processing form: {str(e)}"
    
    return render_template('predict_migraine.html')
