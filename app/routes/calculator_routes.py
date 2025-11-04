from flask import Blueprint, request, render_template, jsonify
from utils.calculator import full_health_calculator
from utils.gemini import get_health_advice  # ✅ Gemini AI integration

calculator_bp = Blueprint('calculator', __name__, url_prefix='/health')


# GET route to show the health calculator form
@calculator_bp.route('/', methods=['GET'])
def show_health_form():
    return render_template('health_form.html')


# POST route for API-based use (e.g., JS fetch)
@calculator_bp.route('/calculate_metrics', methods=['POST'])
def calculate_metrics():
    data = request.json
    result = full_health_calculator(
        age=int(data['age']),
        gender=data['gender'],
        height_cm=float(data['height_cm']),
        weight_kg=float(data['weight_kg']),
        activity_level=data['activity_level'].lower(),
        water_intake_l=float(data['water_intake_l']),
        smokes_per_day=int(data['smokes_per_day'])
    )
    return jsonify(result)


# POST or GET route from form → result page
@calculator_bp.route('/result', methods=['POST', 'GET'])
def show_health_result():
    if request.method == 'POST':
        form = request.form
        gender = form['gender']
        age = int(form['age'])

        # Calculate metrics
        result = full_health_calculator(
            age=age,
            gender=gender,
            height_cm=float(form['height']),
            weight_kg=float(form['weight']),
            activity_level=form['activity'].lower(),
            water_intake_l=2,  # Optional: form.get("water_intake", 2)
            smokes_per_day=0   # Optional: form.get("smokes", 0)
        )

        # Derived metrics
        whr = round(float(form['waist']) / float(form['hip']), 2)
        bp_cat, bp_details = categorize_bp(int(form['systolic']), int(form['diastolic']))

        # 🟠 Health Warnings
        warnings = []

        if result["BMI_Status"] != "Normal":
            warnings.append(f"Your BMI is in the '{result['BMI_Status']}' category. A balanced diet and regular exercise may help.")

        if bp_cat != "Normal":
            warning = f"Your blood pressure falls under '{bp_cat}'."
            if bp_details:
                warning += f" ({bp_details} elevated)"
            warning += " Regular monitoring and a low-sodium diet are advised."
            warnings.append(warning)

        if (gender == 'Male' and whr > 0.90) or (gender == 'Female' and whr > 0.85):
            warnings.append("Your Waist-Hip Ratio suggests a higher cardiovascular risk. Consider reducing abdominal fat.")

        if result["Hydration_Level"] != "Well Hydrated":
            warnings.append(f"Hydration Level: {result['Hydration_Level']}. Increase your water intake for optimal body function.")

        if result["Smoking_Impact"] != "No Impact":
            warnings.append(f"Smoking Impact: {result['Smoking_Impact']}. Reducing or quitting smoking improves overall health.")

        if result["Calorie_Needs"] < 1500:
            warnings.append("Your daily calorie needs are low. Ensure you're not under-eating.")
        elif result["Calorie_Needs"] > 3000:
            warnings.append("Your daily calorie needs are high. Maintain a balanced intake and stay active.")

        # ✅ Gemini AI Health Advice – FIXED to use a dictionary
        advice = get_health_advice({
            "BMI": result["BMI"],
            "BMI_Status": result["BMI_Status"],
            "BMR": result["BMR"],
            "Calorie_Needs": result["Calorie_Needs"],
            "WHR": whr,
            "BP_Category": bp_cat
        })

        # Render final result
        return render_template("health_result.html",
            bmi=result["BMI"],
            bmi_cat=result["BMI_Status"],
            bmr=result["BMR"],
            calorie_needs=result["Calorie_Needs"],
            wh_ratio=whr,
            bp_cat=bp_cat,
            bp_details=bp_details,
            warnings=warnings,
            advice=advice
        )

    # Fallback for GET request
    return render_template("health_form.html")


# 🔧 Blood Pressure Category Logic
def categorize_bp(sys, dia):
    details = []
    
    if sys < 120 and dia < 80:
        return "Normal", None
    
    if sys >= 140 or dia >= 90:
        if sys >= 140:
            details.append(f"Systolic: {sys}")
        if dia >= 90:
            details.append(f"Diastolic: {dia}")
        return "High Blood Pressure (Stage 2)", " & ".join(details)
    
    if 130 <= sys < 140 or 80 <= dia < 90:
        if 130 <= sys < 140:
            details.append(f"Systolic: {sys}")
        if 80 <= dia < 90:
            details.append(f"Diastolic: {dia}")
        return "High Blood Pressure (Stage 1)", " & ".join(details)
    
    if 120 <= sys < 130 and dia < 80:
        return "Elevated", f"Systolic: {sys}"
    
    return "Normal", None










