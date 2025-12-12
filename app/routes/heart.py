from flask import Blueprint, render_template, request
import numpy as np
import pandas as pd
import os

heart_disease_bp = Blueprint('heart_disease', __name__, url_prefix='/heart_disease')

# Load model wrapper (handles inference locally, no API calls)
from app.utils.onnx_inference import get_model

try:
    model_wrapper = get_model('heart')
    print("✅ Heart disease model loaded successfully!")
except Exception as e:
    print(f"⚠️  Heart disease model not found: {e}")
    model_wrapper = None

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

            # Create input dict with feature names
            input_dict = {
                'HighBP': high_bp, 'HighChol': high_chol, 'CholCheck': chol_check,
                'BMI': bmi, 'Smoker': smoker, 'Stroke': stroke, 'Diabetes': diabetes,
                'PhysActivity': phys_activity, 'Fruits': fruits, 'Veggies': veggies,
                'HvyAlcoholConsump': alcohol, 'GenHlth': gen_health, 'MentHlth': ment_health,
                'PhysHlth': phys_health, 'DiffWalk': diff_walk, 'Sex': sex, 'Age': age,
                'Education': education, 'Income': income,
                'Health_Score': health_score, 'Lifestyle_Score': lifestyle_score,
                'BMI_Category_Obese': bmi_cat_obese,
                'BMI_Category_Overweight': bmi_cat_over,
                'BMI_Category_Underweight': bmi_cat_underweight
            }

            # Predict using local model (no API calls)
            prob_array = model_wrapper.predict_proba(input_dict)
            prob = prob_array[0][1]
            prediction = "Yes" if prob > 0.5 else "No"

            return render_template('result_heart.html', 
                                 prediction=prediction, 
                                 probability=f"{prob*100:.2f}",
                                 risk_level="High" if prob > 0.5 else "Low")

        except Exception as e:
            return f"<h3>Error: {e}</h3><p>Please check all fields are filled correctly.</p>"

    return render_template('predict_heart.html')

