import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ✅ Configure Gemini with API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Use gemini-2.5-flash (fast and efficient)
model = genai.GenerativeModel("gemini-2.5-flash")

# ✅ Define health advice prompt function
def get_health_advice(data: dict):
    bmi = data["BMI"]
    bmi_status = data["BMI_Status"]
    bmr = data["BMR"]
    calorie_needs = data["Calorie_Needs"]
    whr = data["WHR"]
    bp_cat = data["BP_Category"]

    prompt = f"""
    Based on the following health profile, provide SHORT and actionable health advice (max 80 words):

    - BMI: {bmi} ({bmi_status})
    - BMR: {bmr} kcal/day
    - Daily Calorie Needs: {calorie_needs} kcal/day
    - Waist-Hip Ratio (WHR): {whr}
    - Blood Pressure Category: {bp_cat}

    Focus ONLY on the top 2-3 most important actions. Be direct and concise.
    
    Format using HTML:
    - Use <strong> for key words
    - Use <ul> and <li> for bullet points (max 3 bullets)
    - Keep each bullet to ONE short sentence
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Unable to generate AI advice at the moment: {str(e)}"



