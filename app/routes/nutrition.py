from flask import Blueprint, render_template, request
from utils.nutrition import (
    search_food_item,
    get_nutrition_details,
    food_benefits_disadvantages
)

nutrition_bp = Blueprint('nutrition', __name__, url_prefix='/nutrition')

@nutrition_bp.route('/', methods=['GET', 'POST'])
def nutrition_lookup():
    if request.method == 'POST':
        food = request.form.get('food')  # ✅ Safe access

        if not food:
            return render_template(
                'nutrition.html',
                nutrition_info={"error": "⚠️ Please enter a food item."}
            )

        ids = search_food_item(food)

        if not ids:
            return render_template(
                'nutrition.html',
                nutrition_info={"error": "❌ No food data found."},
                food_name=food
            )

        fdc_id = ids[0]
        nutrition = get_nutrition_details(fdc_id)

        nutrients = nutrition.get("nutrients", {})  # ✅ Safe fallback

        if not nutrients:
            return render_template(
                'nutrition.html',
                nutrition_info={"error": "❌ No nutrition info found."},
                food_name=food
            )

        info = food_benefits_disadvantages(food)  # ✅ Always returns dict

        nutrition_info = {
            "nutrients": nutrients,
            "benefits": info.get("benefits", []),
            "risks": info.get("disadvantages", []),
            "vitamins": [key for key in nutrients if "Vitamin" in key],
            "minerals": [key for key in nutrients if key in ["Calcium", "Iron", "Magnesium", "Potassium", "Zinc"]],
        }

        return render_template(
            'nutrition.html',
            nutrition_info=nutrition_info,
            food_name=food
        )

    return render_template('nutrition.html')




