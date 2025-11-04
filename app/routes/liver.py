import os
import glob
import joblib
import pandas as pd
from flask import Blueprint, render_template, request

liver_bp = Blueprint('liver', __name__, url_prefix='/liver')

# Load model and preprocessors
def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return max(files, key=os.path.getctime)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'liver')

try:
    model_path = get_latest_file(os.path.join(MODELS_DIR, 'liver_model_*.pkl'))
    scaler_path = get_latest_file(os.path.join(MODELS_DIR, 'scaler_*.pkl'))
    features_path = get_latest_file(os.path.join(MODELS_DIR, 'features_*.pkl'))

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    print("✅ Liver disease model loaded successfully!")
except Exception as e:
    print(f"⚠️  Liver model not found: {e}")
    model = None


@liver_bp.route('/', methods=['GET', 'POST'])
def predict_liver():
    if request.method == 'POST':
        if model is None:
            return "Liver disease model not available", 500
            
        try:
            # Get form data
            form_data = request.form.to_dict()
            
            # Common liver disease indicators (adjust based on actual dataset)
            input_dict = {
                'Age': float(form_data.get('Age', 0)),
                'Gender': 1 if form_data.get('Gender') == 'Male' else 0,
                'Total_Bilirubin': float(form_data.get('Total_Bilirubin', 0)),
                'Direct_Bilirubin': float(form_data.get('Direct_Bilirubin', 0)),
                'Alkaline_Phosphatase': float(form_data.get('Alkaline_Phosphatase', 0)),
                'Alamine_Aminotransferase': float(form_data.get('Alamine_Aminotransferase', 0)),
                'Aspartate_Aminotransferase': float(form_data.get('Aspartate_Aminotransferase', 0)),
                'Total_Protein': float(form_data.get('Total_Protein', 0)),
                'Albumin': float(form_data.get('Albumin', 0)),
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # Add engineered features (same as training)
            if 'SGOT' in input_dict:
                input_df['SGOT_SGPT_Ratio'] = input_df['SGOT'] / (input_df['SGPT'] + 1)
                input_df['High_SGOT_SGPT'] = ((input_df['SGOT'] > 40) & (input_df['SGPT'] > 40)).astype(int)
            
            if 'Albumin' in input_df.columns and 'Total_Protein' in input_df.columns:
                input_df['Albumin_Globulin_Ratio'] = input_df['Albumin'] / (input_df['Total_Protein'] - input_df['Albumin'] + 0.1)
            
            if 'Direct_Bilirubin' in input_df.columns and 'Total_Bilirubin' in input_df.columns:
                input_df['Indirect_Bilirubin'] = input_df['Total_Bilirubin'] - input_df['Direct_Bilirubin']
                input_df['Bilirubin_Ratio'] = input_df['Direct_Bilirubin'] / (input_df['Total_Bilirubin'] + 0.1)
            
            if 'Age' in input_df.columns:
                input_df['Age_Risk'] = pd.cut(input_df['Age'], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3]).astype(int)
                input_df['Age_Squared'] = input_df['Age'] ** 2
            
            # Ensure column order matches training
            input_df = input_df.reindex(columns=feature_names, fill_value=0)
            
            # Scale
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
            
            # Decode prediction
            result = 'Liver Disease Detected' if prediction == 1 else 'No Liver Disease'
            confidence = max(prediction_proba) * 100 if prediction_proba is not None else 0
            
            # Calculate risk level
            if confidence > 80:
                risk_level = 'High' if prediction == 1 else 'Low'
            elif confidence > 60:
                risk_level = 'Moderate'
            else:
                risk_level = 'Uncertain'
            
            return render_template('result_liver.html', 
                                 prediction=result,
                                 confidence=round(confidence, 1),
                                 risk_level=risk_level,
                                 test_results=input_dict)
        
        except Exception as e:
            return f"Error processing form: {str(e)}", 400
    
    return render_template('predict_liver.html')
