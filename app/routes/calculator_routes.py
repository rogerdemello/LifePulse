from flask import Blueprint, request, render_template, jsonify

from app.ml.safety import check_possible
from app.ratelimit import rate_limit
from app.routes.support import FormError, prediction_errors, urgent_interstitial
from app.utils.calculator import full_health_calculator

calculator_bp = Blueprint('calculator', __name__, url_prefix='/health')

# Form field -> the name app.ml.safety uses for its physiological limits and
# red-flag rules. This page collects blood pressure, so it gets the same
# hypertensive-crisis interrupt as the model-backed assessments.
SAFETY_FIELDS = {
    'age': 'Age',
    'systolic': 'Systolic',
    'diastolic': 'Diastolic',
    'water_intake': 'Water Intake',
}


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
        activity_level=data['activity_level'],
        water_intake_l=float(data['water_intake_l']),
        smokes_per_day=int(data['smokes_per_day'])
    )
    return jsonify(result)


# POST or GET route from form → result page
@calculator_bp.route('/result', methods=['POST', 'GET'])
@rate_limit()
@prediction_errors
def show_health_result():
    if request.method == 'POST':
        form = request.form

        required = ['gender', 'age', 'activity', 'height', 'weight', 'waist',
                    'hip', 'systolic', 'diastolic', 'water_intake', 'smokes_per_day']
        missing = [f for f in required if not str(form.get(f, '')).strip()]
        if missing:
            raise FormError(
                "Please fill in every field. Missing: " + ", ".join(sorted(missing))
            )

        # BMI is derived here rather than entered, so add it to what safety sees.
        safety_values = {
            raw: form[field] for field, raw in SAFETY_FIELDS.items()
        }
        height_m = float(form['height']) / 100
        if height_m > 0:
            safety_values['BMI'] = float(form['weight']) / (height_m ** 2)
        check_possible('calculator', safety_values)

        interstitial = urgent_interstitial(safety_values, form)
        if interstitial is not None:
            return interstitial

        gender = form['gender']
        age = int(form['age'])

        # Water intake and cigarettes come from the form now. They used to be
        # hardcoded to 2 litres and 0 cigarettes while the form never asked, so
        # every user in the 57-71 kg band was told "Moderately Hydrated -
        # increase your water intake" based on a number they never entered, and
        # no smoker was ever warned about smoking.
        result = full_health_calculator(
            age=age,
            gender=gender,
            height_cm=float(form['height']),
            weight_kg=float(form['weight']),
            activity_level=form['activity'],
            water_intake_l=float(form['water_intake']),
            smokes_per_day=int(form['smokes_per_day'])
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

        # Returns a list of strings, not a blob of HTML.
        #
        # This used to concatenate <li> tags into a string that the template
        # rendered with |safe. Nothing user-supplied reached it, so it was not
        # exploitable -- but it was the one place in the app where an edit could
        # turn an input into markup, and building HTML in a route is how that
        # edit eventually gets made.
        def lifestyle_tips(metrics, bp_category):
            tips = []
            bmi = metrics.get("BMI", 0)
            calories = metrics.get("Calorie_Needs")

            if bmi >= 30:
                tips.append("Focus on gradual weight loss: balanced diet and "
                            "150+ minutes a week of moderate exercise.")
            elif bmi >= 25:
                tips.append("Aim to reduce weight slightly: combine cardio with "
                            "strength training.")
            else:
                tips.append("Maintain your healthy weight with balanced meals and "
                            "regular activity.")

            if bp_category and bp_category != "Normal":
                tips.append("Reduce sodium, monitor your blood pressure regularly, "
                            "and speak to your doctor if it stays elevated.")

            if calories:
                tips.append(f"Estimated daily calories: {calories:.0f}. Adjust portion "
                            f"sizes for your goals.")

            return tips[:3]

        advice = lifestyle_tips(result, bp_cat)

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










