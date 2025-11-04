import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("USDA_API_KEY")

if not API_KEY:
    print("❗ ERROR: USDA_API_KEY not found. Please add it to your .env file.")

def search_food_item(food_name):
    """
    Search food using USDA API and return a list of matching FDC IDs.
    """
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?api_key={API_KEY}&query={food_name}&pageSize=1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "foods" in data and len(data["foods"]) > 0:
            return [item["fdcId"] for item in data["foods"]]
    except Exception as e:
        print(f"❌ search_food_item() error for '{food_name}':", e)
    return []

def get_nutrition_details(fdc_id):
    """
    Fetch nutrition details by FDC ID and return nutrients dict.
    Handles both flat and nested nutrient formats.
    """
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        nutrients = {}
        for nutrient in data.get("foodNutrients", []):
            # ✅ Handle both USDA formats
            if "nutrientName" in nutrient:
                name = nutrient["nutrientName"]
                value = nutrient.get("value")
                unit = nutrient.get("unitName", "")
            elif "nutrient" in nutrient:
                name = nutrient["nutrient"].get("name")
                value = nutrient.get("amount")
                unit = nutrient.get("unitName", "")
            else:
                continue

            if name and value is not None:
                nutrients[name] = f"{value} {unit}"

        if not nutrients:
            print("⚠️ No nutrients found for FDC ID:", fdc_id)

        return {"nutrients": nutrients}
    except Exception as e:
        print(f"❌ get_nutrition_details() error for ID {fdc_id}:", e)
        return {"nutrients": {}}

def food_benefits_disadvantages(food_name):
    """
    Expanded database of health benefits and considerations for popular foods.
    """
    food_map = {
        "banana": {
            "benefits": [
                "Rich in potassium - supports heart health and regulates blood pressure",
                "Good source of fiber - aids digestion and promotes gut health",
                "Quick energy boost from natural sugars",
                "Contains vitamin B6 for brain health",
                "May improve mood and reduce stress"
            ],
            "disadvantages": [
                "High in sugar - may spike blood sugar in diabetics",
                "Not suitable for low-carb or ketogenic diets",
                "Can cause constipation if eaten unripe"
            ]
        },
        "spinach": {
            "benefits": [
                "Excellent source of iron - prevents anemia",
                "High in vitamins A, C, and K",
                "Rich in antioxidants that fight inflammation",
                "Supports bone health and blood clotting",
                "Low in calories - great for weight management"
            ],
            "disadvantages": [
                "High in oxalates - may contribute to kidney stones",
                "Can interfere with mineral absorption when eaten raw in large amounts",
                "May cause bloating in some people"
            ]
        },
        "apple": {
            "benefits": [
                "High in soluble fiber - lowers cholesterol",
                "Rich in antioxidants - reduces disease risk",
                "Supports heart health and blood sugar regulation",
                "May aid weight loss by promoting fullness",
                "Good for dental health"
            ],
            "disadvantages": [
                "May cause bloating or gas in sensitive individuals",
                "Seeds contain small amounts of cyanide (don't eat in large quantities)",
                "Acidic - may affect tooth enamel"
            ]
        },
        "milk": {
            "benefits": [
                "Excellent source of calcium - builds strong bones",
                "High in protein - supports muscle growth",
                "Contains vitamin D and B vitamins",
                "May improve bone density and reduce fracture risk"
            ],
            "disadvantages": [
                "Lactose intolerance affects many people",
                "Full-fat milk high in saturated fat",
                "May trigger acne in some individuals",
                "Not suitable for vegans"
            ]
        },
        "chicken": {
            "benefits": [
                "High-quality lean protein - builds and repairs muscle",
                "Rich in B vitamins - supports energy metabolism",
                "Good source of selenium - boosts immune function",
                "Low in fat (especially breast meat)",
                "Supports weight management"
            ],
            "disadvantages": [
                "Can harbor harmful bacteria if undercooked",
                "May contain antibiotics or hormones (in factory-farmed)",
                "Skin and dark meat higher in saturated fat"
            ]
        },
        "broccoli": {
            "benefits": [
                "High in vitamin C - boosts immune system",
                "Rich in fiber - supports digestive health",
                "Contains sulforaphane - may have anti-cancer properties",
                "Good source of vitamins K and A",
                "Low calorie - excellent for weight loss"
            ],
            "disadvantages": [
                "May cause gas and bloating",
                "Can interfere with thyroid function if eaten raw in excess",
                "Difficult to digest for some people"
            ]
        },
        "salmon": {
            "benefits": [
                "Excellent source of omega-3 fatty acids - reduces inflammation",
                "High-quality protein - supports muscle health",
                "Rich in vitamin D and B vitamins",
                "Supports heart and brain health",
                "May reduce risk of depression"
            ],
            "disadvantages": [
                "Can be high in mercury (especially farmed)",
                "Expensive compared to other proteins",
                "May contain PCBs or other contaminants",
                "Not suitable for fish allergies"
            ]
        },
        "avocado": {
            "benefits": [
                "Rich in healthy monounsaturated fats",
                "High in potassium - more than bananas",
                "Contains fiber - promotes satiety",
                "Supports heart health and cholesterol levels",
                "Rich in vitamins E and K"
            ],
            "disadvantages": [
                "High in calories - can contribute to weight gain",
                "May cause allergic reactions in some people",
                "Expensive and not always available",
                "Can spoil quickly"
            ]
        },
        "egg": {
            "benefits": [
                "Complete protein source - all essential amino acids",
                "Rich in choline - supports brain function",
                "Contains lutein and zeaxanthin - good for eye health",
                "Inexpensive and versatile",
                "Helps with weight management"
            ],
            "disadvantages": [
                "High in cholesterol (though dietary cholesterol has less impact than once thought)",
                "Common food allergen",
                "Must be cooked properly to avoid salmonella"
            ]
        },
        "rice": {
            "benefits": [
                "Good source of energy from carbohydrates",
                "Gluten-free - safe for celiac disease",
                "Brown rice high in fiber and nutrients",
                "Easy to digest",
                "Affordable and widely available"
            ],
            "disadvantages": [
                "White rice has high glycemic index - spikes blood sugar",
                "Low in protein compared to other grains",
                "Rice may contain arsenic (especially brown rice)",
                "Can contribute to weight gain if eaten in excess"
            ]
        },
        "yogurt": {
            "benefits": [
                "Rich in probiotics - supports gut health",
                "High in protein and calcium",
                "May boost immune function",
                "Supports bone health",
                "Easier to digest than milk for some people"
            ],
            "disadvantages": [
                "Flavored varieties often high in added sugar",
                "Not suitable for lactose intolerance (except lactose-free)",
                "Some brands contain artificial ingredients",
                "Can be high in calories"
            ]
        },
        "oats": {
            "benefits": [
                "High in soluble fiber (beta-glucan) - lowers cholesterol",
                "Stabilizes blood sugar levels",
                "Promotes feelings of fullness - aids weight loss",
                "Rich in antioxidants",
                "Supports heart health"
            ],
            "disadvantages": [
                "May cause bloating or gas in some people",
                "Instant oats often contain added sugars",
                "Can be contaminated with gluten during processing"
            ]
        },
        "tomato": {
            "benefits": [
                "Rich in lycopene - powerful antioxidant",
                "High in vitamin C and potassium",
                "May reduce risk of heart disease and cancer",
                "Supports skin health",
                "Low in calories"
            ],
            "disadvantages": [
                "Acidic - may trigger heartburn or acid reflux",
                "Can cause allergic reactions in some people",
                "Nightshade family - may affect some autoimmune conditions"
            ]
        }
    }
    
    # Normalize the food name for matching
    normalized_name = food_name.lower().strip()
    
    # Try direct match first
    if normalized_name in food_map:
        return food_map[normalized_name]
    
    # Try partial matches
    for key in food_map.keys():
        if key in normalized_name or normalized_name in key:
            return food_map[key]
    
    # Return empty if not found
    return {"benefits": [], "disadvantages": []}






