import math

def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        status = "Underweight"
    elif 18.5 <= bmi < 25:
        status = "Normal"
    elif 25 <= bmi < 30:
        status = "Overweight"
    else:
        status = "Obese"
    return round(bmi, 2), status

def calculate_bmr(gender, age, height_cm, weight_kg):
    if gender.lower() == 'male':
        return round(10 * weight_kg + 6.25 * height_cm - 5 * age + 5, 2)
    else:
        return round(10 * weight_kg + 6.25 * height_cm - 5 * age - 161, 2)

def recommended_water_intake(weight_kg):
    return round(weight_kg * 35 / 1000, 2)  # in liters

def daily_calorie_needs(bmr, activity_level):
    activity_multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9
    }
    return round(bmr * activity_multipliers.get(activity_level, 1.2), 2)

def hydration_score(water_intake_l, recommended_l):
    if water_intake_l >= recommended_l:
        return "Well Hydrated"
    elif water_intake_l >= recommended_l * 0.8:
        return "Moderately Hydrated"
    else:
        return "Low Hydration"

def smoking_impact(smokes_per_day):
    if smokes_per_day == 0:
        return "No Impact"
    elif smokes_per_day <= 5:
        return "Mild"
    elif smokes_per_day <= 10:
        return "Moderate"
    else:
        return "High"

# Example callable function to wrap everything
def full_health_calculator(age, gender, height_cm, weight_kg, activity_level, water_intake_l, smokes_per_day):
    bmi, bmi_status = calculate_bmi(height_cm, weight_kg)
    bmr = calculate_bmr(gender, age, height_cm, weight_kg)
    recommended_water = recommended_water_intake(weight_kg)
    calories = daily_calorie_needs(bmr, activity_level)
    hydration = hydration_score(water_intake_l, recommended_water)
    smoke_effect = smoking_impact(smokes_per_day)

    return {
        "BMI": bmi,
        "BMI_Status": bmi_status,
        "BMR": bmr,
        "Recommended_Water_L": recommended_water,
        "Calorie_Needs": calories,
        "Hydration_Level": hydration,
        "Smoking_Impact": smoke_effect
    }
