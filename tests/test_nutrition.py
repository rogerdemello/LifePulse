"""Nutrition lookup: derived facts, search ranking, and failure handling.

The page used to carry a hand-written table of thirteen foods matched by
substring, which was wrong in both directions -- "chicken nuggets" matched
*chicken* and was described as "low in fat", "eggplant" matched *egg* and was
called a complete protein source, while "oatmeal" matched nothing because the
key was "oats". Everything outside those thirteen returned an empty list with
no indication anything was missing.

Every fact now comes from the numbers USDA returns. No test here touches the
network: the API is mocked so CI stays hermetic and fast.
"""

import pytest

from app.utils import nutrition
from app.utils.nutrition_facts import (
    micronutrients,
    nutrient_facts,
    nutrition_panel,
    summarise,
)

# Per 100 g, in the shape _parse_nutrients produces.
BANANA = {
    1008: {"name": "Energy", "amount": 89.0, "unit": "kcal"},
    1003: {"name": "Protein", "amount": 1.09, "unit": "g"},
    1004: {"name": "Total lipid (fat)", "amount": 0.33, "unit": "g"},
    1258: {"name": "Saturated", "amount": 0.112, "unit": "g"},
    1005: {"name": "Carbohydrate", "amount": 22.84, "unit": "g"},
    1079: {"name": "Fiber", "amount": 2.6, "unit": "g"},
    2000: {"name": "Total Sugars", "amount": 12.23, "unit": "g"},
    1093: {"name": "Sodium", "amount": 1.0, "unit": "mg"},
    1092: {"name": "Potassium", "amount": 358.0, "unit": "mg"},
    1175: {"name": "Vitamin B-6", "amount": 0.367, "unit": "mg"},
    1162: {"name": "Vitamin C", "amount": 8.7, "unit": "mg"},
}

CHEDDAR = {
    1008: {"name": "Energy", "amount": 403.0, "unit": "kcal"},
    1003: {"name": "Protein", "amount": 22.9, "unit": "g"},
    1004: {"name": "Total lipid (fat)", "amount": 33.1, "unit": "g"},
    1258: {"name": "Saturated", "amount": 21.0, "unit": "g"},
    2000: {"name": "Total Sugars", "amount": 0.5, "unit": "g"},
    1093: {"name": "Sodium", "amount": 653.0, "unit": "mg"},
    1087: {"name": "Calcium", "amount": 710.0, "unit": "mg"},
}


# --------------------------------------------------------------------------
# derived facts
# --------------------------------------------------------------------------

def test_high_thresholds_fire_on_a_food_that_deserves_them():
    groups = summarise(nutrient_facts(CHEDDAR))
    cautions = {f.nutrient for f in groups["cautions"]}
    assert {"fat", "saturated fat", "salt"} <= cautions


def test_low_thresholds_fire_on_a_food_that_deserves_them():
    groups = summarise(nutrient_facts(BANANA))
    low = {f.nutrient for f in groups["low_in"]}
    assert {"fat", "saturated fat", "salt"} <= low
    assert not groups["cautions"]


def test_medium_is_reported_rather_than_dropped():
    """Showing only the extremes makes borderline foods look concern-free.

    Chicken nuggets sit just under the FSA fat and salt thresholds; a page that
    listed nothing would imply there was nothing to note.
    """
    nuggets = {
        1004: {"name": "Fat", "amount": 15.42, "unit": "g"},
        1258: {"name": "Saturated", "amount": 3.366, "unit": "g"},
        1093: {"name": "Sodium", "amount": 538.0, "unit": "mg"},
        2000: {"name": "Total Sugars", "amount": 1.27, "unit": "g"},
    }
    moderate = {f.nutrient for f in summarise(nutrient_facts(nuggets))["moderate"]}
    assert {"fat", "saturated fat", "salt"} <= moderate


def test_source_claims_follow_fda_percentages():
    """FDA labelling: 10-19% of the Daily Value is a good source, 20%+ excellent."""
    facts = {f.nutrient: f for f in nutrient_facts(BANANA)}
    # B6: 0.367 mg against a 1.7 mg DV = 21.6% -> excellent
    assert facts["vitamin B6"].level == "excellent"
    # Vitamin C: 8.7 mg against 90 mg = 9.7% -> below the threshold, omitted
    assert "vitamin C" not in facts


def test_every_fact_shows_the_number_behind_it():
    """A label the reader cannot check is no better than the old hardcoded table."""
    for fact in nutrient_facts(CHEDDAR):
        assert fact.detail, f"{fact.text} gives no explanation"
        # The energy fact carries its figure in the headline; the rest justify
        # themselves in the detail, against a stated threshold.
        source = fact.text if fact.kind == "energy" else fact.detail
        assert any(ch.isdigit() for ch in source), fact.text


def test_sodium_is_converted_to_salt():
    """Labelling thresholds are stated in salt; USDA reports sodium."""
    salt_fact = next(f for f in nutrient_facts(CHEDDAR) if f.nutrient == "salt")
    assert "1.63 g salt" in salt_fact.detail
    assert "653" in salt_fact.detail


def test_cautions_are_ordered_before_strengths():
    """Someone checking before an appointment needs the concern first."""
    levels = [f.level for f in nutrient_facts(CHEDDAR)]
    assert levels.index("high") < levels.index("excellent")


def test_missing_nutrients_are_skipped_not_defaulted():
    """A food with no sodium figure must not be called low in salt."""
    facts = nutrient_facts({1008: {"name": "Energy", "amount": 100, "unit": "kcal"}})
    assert not any(f.nutrient == "salt" for f in facts)


# --------------------------------------------------------------------------
# panel and micronutrients
# --------------------------------------------------------------------------

def test_panel_is_ordered_like_a_nutrition_label():
    labels = [row["label"] for row in nutrition_panel(BANANA)]
    assert labels.index("Fat") < labels.index("Carbohydrate") < labels.index("Protein")
    assert labels.index("of which saturates") == labels.index("Fat") + 1


def test_panel_omits_the_long_tail():
    """USDA returns 114 nutrients for a banana, including fluoride."""
    assert len(nutrition_panel(BANANA)) <= 9


def test_micronutrients_are_ranked_and_thresholded():
    rows = micronutrients(BANANA)
    shares = [row["share"] for row in rows]
    assert shares == sorted(shares, reverse=True)
    assert all(row["share"] >= 5 for row in rows)
    # Protein and fibre belong to the main panel, not this list.
    assert not {"Protein", "Fibre"} & {row["label"] for row in rows}


# --------------------------------------------------------------------------
# search ranking
# --------------------------------------------------------------------------

def _food(description, data_type="SR Legacy", fdc_id=1):
    return {"fdcId": fdc_id, "description": description, "dataType": data_type}


@pytest.mark.parametrize("query,candidates,expected", [
    # Relevance must beat brevity: this sent "cheddar cheese" to "Cheese, blue".
    ("cheddar cheese",
     ["Cheese, blue", "Cheese, cheddar", "Cheese, cream"],
     "Cheese, cheddar"),
    ("almond milk",
     ["Milk, whole", "Almond milk, unsweetened", "Milk and cereal bar"],
     "Almond milk, unsweetened"),
    # Brevity still breaks ties between equally relevant entries.
    ("banana",
     ["Bananas, overripe, raw", "Bananas, raw", "Bananas, dehydrated"],
     "Bananas, raw"),
    # Loose matching: singular query, plural record.
    ("lentil", ["Lentils, dry", "Soup, lentil, canned"], "Lentils, dry"),
])
def test_search_ranking(monkeypatch, query, candidates, expected):
    payload = {"foods": [_food(d, fdc_id=i) for i, d in enumerate(candidates)]}
    monkeypatch.setattr(nutrition, "_get", lambda *a, **k: payload)
    assert nutrition.search_foods(query)[0]["description"] == expected


def test_lab_analysed_records_outrank_branded(monkeypatch):
    """A branded "BANANA" listing 12.5 g protein used to be the top hit."""
    payload = {"foods": [
        _food("BANANA", "Branded", 1),
        _food("Bananas, raw", "SR Legacy", 2),
    ]}
    monkeypatch.setattr(nutrition, "_get", lambda *a, **k: payload)
    assert nutrition.search_foods("banana")[0]["description"] == "Bananas, raw"


def test_empty_query_does_not_call_the_api(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the API must not be called for an empty query")

    monkeypatch.setattr(nutrition, "_get", explode)
    assert nutrition.search_foods("   ") == []


# --------------------------------------------------------------------------
# failure handling
# --------------------------------------------------------------------------

def test_a_search_hit_with_no_detail_record_falls_through(monkeypatch):
    """USDA's index contains ids the detail endpoint 404s on.

    "cheddar cheese" hits one. Treating that as a connectivity failure showed
    "could not reach the food database" for a perfectly good search.
    """
    matches = [{"fdc_id": 1, "description": "broken"},
               {"fdc_id": 2, "description": "Cheese, cheddar"}]

    def fake_get_food(fdc_id):
        if fdc_id == 1:
            raise nutrition.FoodNotRetrievable("gone")
        return {"description": "Cheese, cheddar", "nutrients": CHEDDAR}

    monkeypatch.setattr(nutrition, "get_food", fake_get_food)
    food, used = nutrition.get_first_retrievable(matches)
    assert food["description"] == "Cheese, cheddar"
    assert used == 2


def test_all_candidates_unretrievable_returns_nothing(monkeypatch):
    monkeypatch.setattr(nutrition, "get_food", lambda i: (_ for _ in ()).throw(
        nutrition.FoodNotRetrievable("gone")))
    assert nutrition.get_first_retrievable([{"fdc_id": 1, "description": "x"}]) == (None, None)


def test_a_chosen_alternative_is_preferred(monkeypatch):
    matches = [{"fdc_id": 1, "description": "first"},
               {"fdc_id": 9, "description": "chosen"}]
    monkeypatch.setattr(nutrition, "get_food",
                        lambda i: {"description": f"food-{i}", "nutrients": {}})
    _, used = nutrition.get_first_retrievable(matches, preferred_id="9")
    assert used == 9


def test_missing_api_key_is_reported_not_silently_empty(monkeypatch):
    monkeypatch.setattr(nutrition, "API_KEY", None)
    with pytest.raises(nutrition.NutritionUnavailable, match="USDA_API_KEY"):
        nutrition.search_foods("banana")


def test_page_says_so_when_the_key_is_missing(client, monkeypatch):
    monkeypatch.setattr("app.routes.nutrition.is_configured", lambda: False)
    body = client.get("/nutrition/").get_data(as_text=True)
    assert "isn't configured" in body
    assert "USDA_API_KEY" in body


# --------------------------------------------------------------------------
# route behaviour
# --------------------------------------------------------------------------

def test_blank_search_is_rejected(client):
    response = client.post("/nutrition/", data={"food": "   "})
    assert response.status_code == 400
    assert "Please enter a food" in response.get_data(as_text=True)


def test_no_match_is_a_404_not_an_error(client, monkeypatch):
    monkeypatch.setattr("app.routes.nutrition.search_foods", lambda q, **k: [])
    response = client.post("/nutrition/", data={"food": "zzzznotafood"})
    assert response.status_code == 404
    assert "No food matching" in response.get_data(as_text=True)


def test_a_lookup_renders_facts_and_the_panel(client, monkeypatch):
    monkeypatch.setattr(
        "app.routes.nutrition.search_foods",
        lambda q, **k: [{"fdc_id": 1, "description": "Cheese, cheddar",
                         "data_type": "SR Legacy"}],
    )
    monkeypatch.setattr(
        "app.routes.nutrition.get_first_retrievable",
        lambda m, preferred_id=None: (
            {"fdc_id": 1, "description": "Cheese, cheddar",
             "data_type": "SR Legacy", "brand": None, "nutrients": CHEDDAR}, 1),
    )
    body = client.post("/nutrition/", data={"food": "cheddar"}).get_data(as_text=True)

    assert "Cheese, cheddar" in body
    assert "High in saturated fat" in body
    assert "Excellent source of calcium" in body
    assert "Where those come from" in body      # every label is justified
    assert "adjusted for your age" in body      # and scoped honestly


def test_upstream_failure_is_a_503_with_a_readable_message(client, monkeypatch):
    def unavailable(*args, **kwargs):
        raise nutrition.NutritionUnavailable("USDA is down for maintenance.")

    monkeypatch.setattr("app.routes.nutrition.search_foods", unavailable)
    response = client.post("/nutrition/", data={"food": "banana"})
    assert response.status_code == 503
    body = response.get_data(as_text=True)
    assert "USDA is down for maintenance." in body
    assert "Traceback" not in body
