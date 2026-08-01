"""Turn USDA nutrient values into statements about a food.

This replaces a hand-written table of thirteen foods matched by substring. That
table was wrong in both directions: "eggplant" matched *egg* and was told it was
a complete protein source, "almond milk" and "milkshake" both matched *milk*,
and "chicken nuggets" matched *chicken* and was described as "low in fat". Foods
that were in the table but spelled differently -- "oatmeal" against a key of
"oats" -- got nothing at all. Everything outside those thirteen silently
returned an empty list, so most searches showed no health information and no
indication that any was missing.

Everything here is derived from the numbers USDA actually returns, so it covers
every food in the database and can be checked against the nutrient panel shown
alongside it. Two published systems, both designed for exactly this:

* **UK FSA front-of-pack thresholds** for fat, saturates, sugars and salt.
  Defined per 100 g, which is the basis USDA reports on, so no conversion or
  serving-size guesswork is involved.
* **US FDA Daily Values** for "source of" claims. FDA labelling calls 10-19%
  of the DV a "good source" and 20% or more an "excellent source".

Both are consumer-facing labelling standards, not clinical advice, and the page
says so. Nothing here tells anyone what to eat.
"""

from __future__ import annotations

from dataclasses import dataclass

# USDA nutrient IDs. Stable, unlike the display names -- the same nutrient comes
# back as "Total Sugars" from one endpoint and "Sugars, total including NLEA"
# from another, which is what the old name-matching parser tripped over.
ENERGY = 1008
PROTEIN = 1003
FAT = 1004
SATURATES = 1258
CARBS = 1005
FIBRE = 1079
SUGARS = 2000
SODIUM = 1093
CHOLESTEROL = 1253

# (nutrient id, label, Daily Value, unit)
DAILY_VALUES = [
    (PROTEIN, "protein", 50, "g"),
    (FIBRE, "fibre", 28, "g"),
    (1087, "calcium", 1300, "mg"),
    (1089, "iron", 18, "mg"),
    (1090, "magnesium", 420, "mg"),
    (1092, "potassium", 4700, "mg"),
    (1095, "zinc", 11, "mg"),
    (1091, "phosphorus", 1250, "mg"),
    (1162, "vitamin C", 90, "mg"),
    (1106, "vitamin A", 900, "µg"),
    (1109, "vitamin E", 15, "mg"),
    (1114, "vitamin D", 20, "µg"),
    (1185, "vitamin K", 120, "µg"),
    (1165, "thiamin", 1.2, "mg"),
    (1166, "riboflavin", 1.3, "mg"),
    (1167, "niacin", 16, "mg"),
    (1175, "vitamin B6", 1.7, "mg"),
    (1177, "folate", 400, "µg"),
    (1178, "vitamin B12", 2.4, "µg"),
]

# UK FSA front-of-pack thresholds, g per 100 g.
# (nutrient id, label, low limit, high limit)
TRAFFIC_LIGHTS = [
    (FAT, "fat", 3.0, 17.5),
    (SATURATES, "saturated fat", 1.5, 5.0),
    (SUGARS, "sugars", 5.0, 22.5),
]

SALT_LOW, SALT_HIGH = 0.3, 1.5

# Sodium (mg) to salt (g). Salt is sodium chloride: 1 g sodium ≈ 2.5 g salt.
SODIUM_MG_TO_SALT_G = 2.5 / 1000


@dataclass(frozen=True)
class Fact:
    """One statement about a food, with the number it came from."""

    kind: str        # "source" | "high" | "low" | "energy"
    level: str       # "good" | "excellent" | "high" | "low" | "medium" | "info"
    nutrient: str
    text: str
    detail: str

    @property
    def is_caution(self):
        return self.level == "high"

    @property
    def is_positive(self):
        return self.level in ("good", "excellent", "low")


def _amount(nutrients, nutrient_id):
    """Amount per 100 g for a nutrient id, or None."""
    entry = nutrients.get(nutrient_id)
    if not entry:
        return None
    value = entry.get("amount")
    return None if value is None else float(value)


def nutrient_facts(nutrients):
    """Derive the labelled facts for a food from its per-100 g nutrients.

    ``nutrients`` maps USDA nutrient id -> {"name", "amount", "unit"}.
    Returns cautions first, then positives -- a food being high in salt matters
    more to someone checking before a doctor's appointment than it being a
    source of riboflavin.
    """
    facts = []

    energy = _amount(nutrients, ENERGY)
    if energy is not None:
        facts.append(Fact(
            kind="energy", level="info", nutrient="energy",
            text=f"{energy:g} kcal per 100 g",
            detail="Energy density, not a judgement — an avocado and a biscuit "
                   "can be similar here for very different reasons.",
        ))

    for nutrient_id, label, low, high in TRAFFIC_LIGHTS:
        value = _amount(nutrients, nutrient_id)
        if value is None:
            continue
        if value > high:
            facts.append(Fact(
                kind="high", level="high", nutrient=label,
                text=f"High in {label}",
                detail=f"{value:g} g per 100 g. The UK front-of-pack threshold "
                       f"for high {label} is more than {high:g} g.",
            ))
        elif value <= low:
            facts.append(Fact(
                kind="low", level="low", nutrient=label,
                text=f"Low in {label}",
                detail=f"{value:g} g per 100 g, at or below the {low:g} g "
                       f"threshold for low {label}.",
            ))
        else:
            facts.append(Fact(
                kind="medium", level="medium", nutrient=label,
                text=f"Medium {label}",
                detail=f"{value:g} g per 100 g, between the {low:g} g and "
                       f"{high:g} g thresholds.",
            ))

    sodium = _amount(nutrients, SODIUM)
    if sodium is not None:
        salt = sodium * SODIUM_MG_TO_SALT_G
        if salt > SALT_HIGH:
            level, text = "high", "High in salt"
        elif salt <= SALT_LOW:
            level, text = "low", "Low in salt"
        else:
            level, text = "medium", "Medium salt"
        facts.append(Fact(
            kind=level, level=level, nutrient="salt", text=text,
            detail=f"{salt:.2f} g salt per 100 g ({sodium:g} mg sodium). "
                   f"High is above {SALT_HIGH:g} g, low is {SALT_LOW:g} g or less.",
        ))

    for nutrient_id, label, daily_value, unit in DAILY_VALUES:
        value = _amount(nutrients, nutrient_id)
        if value is None or value <= 0:
            continue
        share = 100 * value / daily_value
        if share >= 20:
            level, prefix = "excellent", "Excellent source of"
        elif share >= 10:
            level, prefix = "good", "Good source of"
        else:
            continue
        facts.append(Fact(
            kind="source", level=level, nutrient=label,
            text=f"{prefix} {label}",
            detail=f"{value:g} {unit} per 100 g — about {share:.0f}% of the "
                   f"US Daily Value ({daily_value:g} {unit}).",
        ))

    order = {"high": 0, "excellent": 1, "good": 2, "low": 3, "medium": 4, "info": 5}
    facts.sort(key=lambda f: order.get(f.level, 9))
    return facts


def summarise(facts):
    """Split derived facts into the groups the page renders.

    ``moderate`` is shown rather than dropped. Chicken nuggets come in just
    under the FSA thresholds for fat and salt, so showing only the extremes
    would present them as free of concerns. "Medium salt" is a real answer.
    """
    return {
        "cautions": [f for f in facts if f.level == "high"],
        "strengths": [f for f in facts if f.level in ("excellent", "good")],
        "moderate": [f for f in facts if f.level == "medium"],
        "low_in": [f for f in facts if f.level == "low"],
        "energy": next((f for f in facts if f.kind == "energy"), None),
    }


# Nutrients worth showing in the panel, in the order a label would list them.
PANEL = [
    (ENERGY, "Energy"),
    (FAT, "Fat"),
    (SATURATES, "of which saturates"),
    (CARBS, "Carbohydrate"),
    (SUGARS, "of which sugars"),
    (FIBRE, "Fibre"),
    (PROTEIN, "Protein"),
    (SODIUM, "Sodium"),
    (CHOLESTEROL, "Cholesterol"),
]


def nutrition_panel(nutrients):
    """The core nutrients, ordered like a nutrition label.

    The page used to dump every nutrient USDA returned -- 114 rows for a banana,
    including fluoride and individual sugars -- with no ordering.
    """
    rows = []
    for nutrient_id, label in PANEL:
        entry = nutrients.get(nutrient_id)
        if not entry or entry.get("amount") is None:
            continue
        rows.append({
            "label": label,
            "amount": entry["amount"],
            "unit": entry.get("unit", ""),
            "indent": label.startswith("of which"),
        })
    return rows


def portion_rows(nutrients, portions):
    """What a realistic helping actually contains.

    Per-100 g is the right basis for comparing two foods and the wrong one for
    answering "I ate a banana". USDA supplies household measures, so each is
    scaled here: energy, and the three nutrients people are usually watching.
    """
    energy = _amount(nutrients, ENERGY)
    sugars = _amount(nutrients, SUGARS)
    saturates = _amount(nutrients, SATURATES)
    sodium = _amount(nutrients, SODIUM)

    rows = []
    for portion in portions or []:
        factor = portion["grams"] / 100.0
        rows.append({
            "label": portion["label"],
            "grams": portion["grams"],
            "energy": None if energy is None else round(energy * factor),
            "sugars": None if sugars is None else round(sugars * factor, 1),
            "saturates": None if saturates is None else round(saturates * factor, 1),
            "salt": None if sodium is None
                    else round(sodium * factor * SODIUM_MG_TO_SALT_G, 2),
        })
    return rows


def micronutrients(nutrients, minimum_share=5.0):
    """Vitamins and minerals present at a meaningful level, richest first.

    Anything under 5% of the Daily Value per 100 g is noise on a label.
    """
    rows = []
    for nutrient_id, label, daily_value, unit in DAILY_VALUES:
        if nutrient_id in (PROTEIN, FIBRE):
            continue  # already in the main panel
        value = _amount(nutrients, nutrient_id)
        if value is None or value <= 0:
            continue
        share = 100 * value / daily_value
        if share < minimum_share:
            continue
        rows.append({
            "label": label.capitalize(),
            "amount": value,
            "unit": unit,
            "share": round(share),
        })
    rows.sort(key=lambda r: r["share"], reverse=True)
    return rows
