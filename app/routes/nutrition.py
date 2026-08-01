"""Food lookup against USDA FoodData Central."""

import logging

from flask import Blueprint, render_template, request

from app.ratelimit import rate_limit
from app.utils.nutrition import (
    NutritionUnavailable,
    get_first_retrievable,
    is_configured,
    search_foods,
)
from app.utils.nutrition_facts import (
    micronutrients,
    nutrient_facts,
    nutrition_panel,
    summarise,
)

log = logging.getLogger(__name__)

nutrition_bp = Blueprint("nutrition", __name__, url_prefix="/nutrition")


@nutrition_bp.route("/", methods=["GET", "POST"])
# Every lookup makes two calls to a third-party API on a shared key. Throttling
# here protects the key's quota as much as this server.
@rate_limit(limit=20, window=60)
def nutrition_lookup():
    if request.method != "POST":
        return render_template("nutrition.html", configured=is_configured())

    query = (request.form.get("food") or "").strip()
    if not query:
        return render_template(
            "nutrition.html", configured=is_configured(),
            error="Please enter a food to look up.",
        ), 400

    # A specific result was chosen from the alternatives on a previous search.
    chosen = request.form.get("fdc_id")

    try:
        matches = search_foods(query)
        if not matches:
            return render_template(
                "nutrition.html", configured=True, food_name=query,
                error=f"No food matching “{query}” was found in the USDA database.",
            ), 404

        food, selected_id = get_first_retrievable(matches, preferred_id=chosen)
        if food is None:
            return render_template(
                "nutrition.html", configured=True, food_name=query,
                error=f"USDA lists matches for “{query}” but has no full "
                      f"nutrient record for any of them. Try a simpler name.",
            ), 404
    except NutritionUnavailable as exc:
        log.warning("nutrition lookup failed for %r: %s", query, exc)
        return render_template(
            "nutrition.html", configured=is_configured(), food_name=query,
            error=str(exc),
        ), 503

    facts = nutrient_facts(food["nutrients"])

    return render_template(
        "nutrition.html",
        configured=True,
        food_name=query,
        food=food,
        facts=facts,
        groups=summarise(facts),
        panel=nutrition_panel(food["nutrients"]),
        micros=micronutrients(food["nutrients"]),
        alternatives=[m for m in matches if str(m["fdc_id"]) != str(selected_id)],
    )
