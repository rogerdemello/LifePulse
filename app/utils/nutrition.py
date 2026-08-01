"""USDA FoodData Central client.

Two things this fixes beyond tidying.

**It asked for one result and took it.** ``pageSize=1`` with no data-type filter
meant a search for "banana" returned a *branded* product also called BANANA,
listing 12.5 g of protein per 100 g. A banana has about 1.1 g. Whatever a
manufacturer happened to submit outranked the laboratory-analysed entry. Results
are now ranked, generic whole-food records are preferred, and the alternatives
are offered so the user can pick.

**It matched nutrients by display name.** The same nutrient arrives as "Total
Sugars" from one endpoint and "Sugars, total including NLEA" from another.
Everything here keys on USDA's stable numeric nutrient ids instead.
"""

from __future__ import annotations

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

API_KEY = os.getenv("USDA_API_KEY")
BASE_URL = "https://api.nal.usda.gov/fdc/v1"
TIMEOUT = 10

# Laboratory-analysed generic foods first, manufacturer submissions last.
# Branded entries are self-reported and vary wildly in quality.
#
# Foundation and SR Legacy share a tier deliberately. Both are lab-analysed, so
# separating them lets a more specific record outrank a more general one:
# "Bananas, overripe, raw" (Foundation) would beat "Bananas, raw" (SR Legacy)
# for a search of "banana", which is not what anyone means.
DATA_TYPE_RANK = {
    "Foundation": 0,
    "SR Legacy": 0,
    "Survey (FNDDS)": 1,
    "Branded": 2,
}

# Only parenthesis-free names are sent as a filter. Including "Survey (FNDDS)"
# makes USDA's edge proxy return an HTML 400 rather than a JSON error, and
# intermittently, so it presents as flaky connectivity rather than a bad
# request. Survey and Branded records still surface through the unfiltered
# fallback in search_foods and are ordered correctly by _rank -- they simply
# are not requested by name.
PREFERRED_DATA_TYPES = ["Foundation", "SR Legacy"]


class NutritionUnavailable(RuntimeError):
    """The lookup cannot run — no API key, or USDA is unreachable."""


class FoodNotRetrievable(NutritionUnavailable):
    """A search hit exists but its full record cannot be fetched.

    USDA's search index returns ids that the detail endpoint 404s on. A search
    for "cheddar cheese" hits one. Treating that as a connectivity failure would
    show "could not reach the food database" for a perfectly good search, so it
    is distinguished here and the caller falls through to the next candidate.
    """


def is_configured():
    return bool(API_KEY)


def _get(path, **params):
    if not API_KEY:
        raise NutritionUnavailable(
            "USDA_API_KEY is not set, so food lookups are unavailable. "
            "A free key is available at fdc.nal.usda.gov/api-key-signup.html"
        )
    try:
        response = requests.get(
            f"{BASE_URL}/{path}", params={"api_key": API_KEY, **params},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise NutritionUnavailable(
            "The USDA food database did not respond in time. Please try again."
        ) from exc
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 404:
            raise FoodNotRetrievable(
                "That food's full record is not available from USDA."
            ) from exc
        if status == 429:
            raise NutritionUnavailable(
                "The USDA food database is rate-limiting this server. "
                "Please try again in a few minutes."
            ) from exc
        raise NutritionUnavailable(
            "Could not reach the USDA food database. Please try again."
        ) from exc
    except ValueError as exc:
        raise NutritionUnavailable(
            "The USDA food database returned an unreadable response."
        ) from exc


def _matches(word, query_word):
    """Loose word match, so "banana" finds "Bananas" and "lentil" finds "Lentils"."""
    return word.startswith(query_word) or query_word.startswith(word)


def _rank(food, query):
    """Sort key: trustworthy data type, then relevance, then the most generic entry.

    Relevance has to come before brevity. Ranking on description length alone
    sent "cheddar cheese" to "Cheese, blue" and "almond milk" to "Milk and
    cereal bar" -- both shorter than the right answer, and both wrong. Counting
    how many of the query's words actually appear fixes that, while brevity
    still breaks ties so "banana" lands on "Bananas, raw" rather than
    "Bananas, overripe, raw".
    """
    description = (food.get("description") or "").lower()
    words = description.replace(",", " ").split()
    query_words = [w for w in query.lower().split() if w]

    matched = sum(
        1 for q in query_words if any(_matches(w, q) for w in words)
    )
    leads = words and query_words and any(_matches(words[0], q) for q in query_words)

    return (
        DATA_TYPE_RANK.get(food.get("dataType"), 9),
        -matched,                       # most query words present wins
        0 if leads else 1,              # then: is it the head noun?
        description.count(","),         # then: fewest qualifying clauses
        len(description),
    )


def search_foods(query, limit=6):
    """Return ranked candidate foods for ``query``.

    Each item is ``{fdc_id, description, data_type}``. Empty list if nothing
    matched — that is a real answer, not an error.
    """
    query = (query or "").strip()
    if not query:
        return []

    payload = _get(
        "foods/search",
        query=query,
        pageSize=25,
        dataType=PREFERRED_DATA_TYPES,
    )
    foods = payload.get("foods") or []

    if not foods:
        # Nothing lab-analysed matched; fall back to the full index, which
        # brings in Survey and Branded records. _rank still puts them last.
        payload = _get("foods/search", query=query, pageSize=25)
        foods = payload.get("foods") or []

    foods.sort(key=lambda food: _rank(food, query))
    return [
        {
            "fdc_id": food["fdcId"],
            "description": food.get("description", "Unknown food"),
            "data_type": food.get("dataType", ""),
        }
        for food in foods[:limit]
    ]


def _parse_nutrients(payload):
    """Map USDA nutrient id -> {name, amount, unit}, per 100 g.

    Handles both response shapes: the detail endpoint nests the nutrient under
    a "nutrient" key, the search endpoint flattens it.
    """
    nutrients = {}
    for entry in payload.get("foodNutrients", []) or []:
        if "nutrient" in entry:
            nutrient = entry["nutrient"] or {}
            nutrient_id = nutrient.get("id")
            name = nutrient.get("name")
            unit = nutrient.get("unitName")
            amount = entry.get("amount")
        else:
            nutrient_id = entry.get("nutrientId")
            name = entry.get("nutrientName")
            unit = entry.get("unitName")
            amount = entry.get("value")

        if nutrient_id is None or amount is None:
            continue

        # Energy is reported twice, in kcal and kJ, under different ids; the
        # panel wants kcal, which is id 1008.
        nutrients[int(nutrient_id)] = {
            "name": name,
            "amount": amount,
            "unit": (unit or "").replace("KCAL", "kcal").replace("G", "g")
                                 .replace("MG", "mg").replace("UG", "µg"),
        }
    return nutrients


def _parse_portions(payload, limit=5):
    """Household measures for a food, as ``{label, grams}``.

    Everything else on the page is per 100 g, which is the right basis for
    comparing two foods and the wrong one for "I ate a banana". USDA knows a
    large banana is 136 g; there is no reason to make someone guess.

    ``NLEA serving`` is the serving size used on nutrition labels, so it leads
    where present.
    """
    portions = []
    for entry in payload.get("foodPortions", []) or []:
        grams = entry.get("gramWeight")
        if not grams:
            continue
        modifier = (entry.get("modifier") or "").strip()
        unit = (entry.get("measureUnit") or {}).get("name") or ""
        if unit in ("undetermined", ""):
            unit = ""
        amount = entry.get("amount")

        label = " ".join(part for part in [
            f"{amount:g}" if amount and amount != 1 else "",
            unit,
            modifier,
        ] if part).strip()
        if not label:
            continue
        portions.append({"label": label, "grams": float(grams)})

    def rank(portion):
        label = portion["label"].lower()
        return (0 if "nlea" in label else 1, portion["grams"])

    portions.sort(key=rank)

    seen, unique = set(), []
    for portion in portions:
        key = round(portion["grams"], 1)
        if key in seen:
            continue
        seen.add(key)
        # "NLEA serving" is jargon; it means the labelling serving size.
        if "nlea" in portion["label"].lower():
            portion = {**portion, "label": "standard serving"}
        unique.append(portion)

    return unique[:limit]


def get_food(fdc_id):
    """Full record for one food: description, data type, nutrients and portions."""
    payload = _get(f"food/{fdc_id}")
    return {
        "fdc_id": fdc_id,
        "description": payload.get("description", "Unknown food"),
        "data_type": payload.get("dataType", ""),
        "brand": payload.get("brandOwner"),
        "nutrients": _parse_nutrients(payload),
        "portions": _parse_portions(payload),
    }


def get_first_retrievable(matches, preferred_id=None):
    """Fetch the best match whose full record actually exists.

    Returns ``(food, used_id)``, or ``(None, None)`` if every candidate 404s.
    Walking the list matters because USDA's index contains ids the detail
    endpoint does not serve.
    """
    ordered = list(matches)
    if preferred_id is not None:
        ordered.sort(key=lambda m: str(m["fdc_id"]) != str(preferred_id))

    for match in ordered:
        try:
            return get_food(match["fdc_id"]), match["fdc_id"]
        except FoodNotRetrievable:
            log.info("USDA has no detail record for %s (%s)",
                     match["fdc_id"], match.get("description"))
            continue
    return None, None
