import os
import pickle
import pandas as pd
import joblib
from flask import Blueprint, render_template, request

migraine_bp = Blueprint('migraine', __name__, url_prefix='/migraine')

# Load model wrapper (local inference, no API)
from app.utils.onnx_inference import get_model

try:
    model_wrapper = get_model('migraine')
    print("✅ Migraine model loaded successfully!")
except Exception as e:
    print(f"⚠️  Migraine model not found: {e}")
    model_wrapper = None


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
            
            # Get features expected by model
            feature_list = model_wrapper.features
            
            # Ensure all features exist and order matches
            input_df = input_df.reindex(columns=feature_list, fill_value=0)
            
            # Predict using local model (no API calls)
            prediction = model_wrapper.predict(input_df.iloc[0].to_dict())[0]
            prediction_proba = model_wrapper.predict_proba(input_df.iloc[0].to_dict())[0]
            
            # Decode prediction
            result = 'Migraine Risk' if prediction == 1 else 'No Migraine Risk'
            confidence = max(prediction_proba) * 100
            
            return render_template('result_migraine.html', 
                                 prediction=result,
                                 confidence=round(confidence, 1))
        
        except Exception as e:
            return f"Error processing form: {str(e)}"
    
    return render_template('predict_migraine.html')
