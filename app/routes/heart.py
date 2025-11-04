from flask import Blueprint, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

heart_disease_bp = Blueprint('heart_disease', __name__, url_prefix='/heart_disease')

# ✅ Use absolute path for model directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # routes/
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))  # app/
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'heart')  # app/models/heart

def load_latest_file(prefix):
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"❌ Model folder not found: {MODEL_DIR}")

    files = sorted([f for f in os.listdir(MODEL_DIR) if prefix in f], reverse=True)
    
    if not files:
        raise FileNotFoundError(f"❌ No file found with prefix '{prefix}' in {MODEL_DIR}")

    latest_file = files[0]
    return joblib.load(os.path.join(MODEL_DIR, latest_file))

# ✅ Load model and scaler
model = load_latest_file("heart_disease_model")
scaler = load_latest_file("scaler")

@heart_disease_bp.route('/', methods=['GET', 'POST'])
def predict_heart_disease():
    if request.method == 'POST':
        try:
            # ✅ Collect ALL form inputs
            high_bp = int(request.form['high_bp'])
            high_chol = int(request.form['high_chol'])
            chol_check = int(request.form['chol_check'])
            bmi = float(request.form['bmi'])
            smoker = int(request.form['smoker'])
            stroke = int(request.form['stroke'])
            diabetes = int(request.form['diabetes'])
            phys_activity = int(request.form['phys_activity'])
            fruits = int(request.form['fruit'])
            veggies = int(request.form['veggies'])
            alcohol = int(request.form['alcohol'])
            gen_health = int(request.form['gen_health'])
            ment_health = int(request.form['ment_health'])
            phys_health = int(request.form['phys_health'])
            diff_walk = int(request.form['diff_walk'])
            sex = int(request.form['sex'])
            age = float(request.form['age'])
            
            # ✅ Set default values for removed fields
            education = 4  # Default: High school graduate
            income = 5     # Default: Middle income ($25k-$35k)

            # ✅ Derived Features (same as training)
            health_score = gen_health + phys_health + ment_health
            lifestyle_score = phys_activity + fruits + veggies - smoker - alcohol

            # BMI Categories (drop_first=True drops 'Normal' alphabetically)
            bmi_cat_obese = int(bmi >= 30)
            bmi_cat_over = int(25 <= bmi < 30)
            bmi_cat_underweight = int(bmi < 18.5)

            # ✅ Create DataFrame with EXACT column order as training
            input_data = pd.DataFrame([[  
                high_bp, high_chol, chol_check, bmi, smoker, stroke, diabetes,
                phys_activity, fruits, veggies, alcohol, gen_health, ment_health,
                phys_health, diff_walk, sex, age, education, income,
                health_score, lifestyle_score,
                bmi_cat_obese, bmi_cat_over, bmi_cat_underweight
            ]], columns=[
                'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes',
                'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'GenHlth', 'MentHlth',
                'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
                'Health_Score', 'Lifestyle_Score',
                'BMI_Category_Obese', 'BMI_Category_Overweight', 'BMI_Category_Underweight'
            ])

            # ✅ Predict
            input_scaled = scaler.transform(input_data)
            prob = model.predict_proba(input_scaled)[0][1]
            prediction = "Yes" if prob > 0.5 else "No"

            return render_template('result_heart.html', 
                                 prediction=prediction, 
                                 probability=f"{prob*100:.2f}",
                                 risk_level="High" if prob > 0.5 else "Low")

        except Exception as e:
            return f"<h3>Error: {e}</h3><p>Please check all fields are filled correctly.</p>"

    return render_template('predict_heart.html')

