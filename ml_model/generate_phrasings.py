"""Pre-write the result copy with Azure OpenAI, once, offline.

    python ml_model/generate_phrasings.py            # writes app/ml/phrasings.json
    python ml_model/generate_phrasings.py --dry-run  # show what would be sent
    python ml_model/generate_phrasings.py --check    # verify the committed file

Better wording for result explanations and doctor questions was wanted, but
generating it per request would mean sending each user's result to Azure -- and
/privacy promises that assessment answers never leave this server.

They don't have to. The *shape* of every sentence is knowable in advance: there
are only so many (field, direction) pairs and so many result bands. This walks
that space once, on a developer's machine with no user involved, and commits
the output. At runtime the app picks a sentence out of the file and fills in the
person's own numbers locally.

So the copy is LLM-written and nothing about anyone is ever transmitted. The
tradeoff is that the wording cannot react to an individual -- which for
generic health phrasing is not much of a loss, and is what makes the privacy
claim survivable.

Requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and
AZURE_OPENAI_DEPLOYMENT. The committed file means nobody else needs them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.azure_openai import AzureUnavailable, build_request, complete_json, is_configured
from app.ml.features import FIELD_LABELS, HEART_RAW, MIGRAINE_RAW
from app.ml.lifestyle import FACTORS

OUT = ROOT / "app" / "ml" / "phrasings.json"

log = logging.getLogger("phrasings")

SYSTEM = (
    "You write short, plain sentences for a health screening tool that people "
    "read before deciding whether to see a doctor. Rules: British English; "
    "grade-8 reading level; never diagnose; never instruct someone to start or "
    "stop a treatment; never promise an outcome; no exclamation marks; no "
    "reassurance that could stop someone seeking care. Reply with JSON only."
)

# The full space of sentences the result pages need. Small and enumerable,
# which is the whole reason this can be done ahead of time.
DIRECTIONS = ("raised", "lowered")


def _explanation_prompt(field, direction):
    label = FIELD_LABELS.get(field, field)
    return (
        f"A screening model found that a person's answer for '{label}' "
        f"{direction} their estimated risk relative to a typical respondent.\n\n"
        f"Write one sentence, at most 20 words, explaining what that means to "
        f"them. Do not include any number -- the app fills those in. Do not "
        f"tell them what to do about it.\n\n"
        f'Reply as {{"sentence": "..."}}'
    )


def _questions_prompt(topic, band):
    return (
        f"A person has just been shown a {topic} screening result in the "
        f"'{band}' range. Write 3 questions they could ask their doctor about "
        f"it.\n\nEach must be a question the person asks, never advice or an "
        f"instruction. At most 25 words each. Do not assume a diagnosis.\n\n"
        f'Reply as {{"questions": ["...", "...", "..."]}}'
    )


JOBS = []
for _field in dict.fromkeys(list(HEART_RAW) + list(MIGRAINE_RAW)):
    for _direction in DIRECTIONS:
        JOBS.append(("explanation", f"{_field}|{_direction}",
                     _explanation_prompt(_field, _direction)))
for _topic, _bands in [
    ("heart disease risk", ["low", "raised", "high"]),
    ("migraine risk", ["low", "raised"]),
    ("sleep apnea", ["low", "raised", "high"]),
    ("lifestyle", ["strong", "mixed", "needs work"]),
]:
    for _band in _bands:
        JOBS.append(("questions", f"{_topic}|{_band}",
                     _questions_prompt(_topic, _band)))


def _banned(text):
    """Reject copy that crosses the lines the system prompt sets.

    A model that ignores an instruction is an ordinary occurrence, and this
    text goes in front of people deciding whether to seek care. Anything that
    instructs or reassures is dropped rather than committed.
    """
    lowered = text.lower()
    return [phrase for phrase in (
        "you should", "you must", "make sure you", "don't worry",
        "no need to", "nothing to worry", "you have ", "diagnos",
        "stop taking", "start taking", "guarantee",
    ) if phrase in lowered]


def generate():
    if not is_configured():
        raise SystemExit(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT.\n"
            "The committed phrasings.json means this is only needed to "
            "regenerate the copy."
        )

    output = {"explanation": {}, "questions": {}}
    rejected = []

    for kind, key, prompt in JOBS:
        try:
            payload = complete_json(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": prompt}],
                max_tokens=220,
            )
        except AzureUnavailable as exc:
            log.warning("  %s %s: %s", kind, key, exc)
            continue

        if kind == "explanation":
            sentence = str(payload.get("sentence", "")).strip()
            problems = _banned(sentence)
            if not sentence or problems:
                rejected.append((key, problems or ["empty"]))
                continue
            output["explanation"][key] = sentence
        else:
            questions = [str(q).strip() for q in payload.get("questions", [])]
            kept = [q for q in questions if q.endswith("?") and not _banned(q)]
            if not kept:
                rejected.append((key, ["no usable questions"]))
                continue
            output["questions"][key] = kept[:3]

        log.info("  %s %s", kind, key)

    return output, rejected


def check():
    """Verify the committed file without calling Azure."""
    if not OUT.exists():
        raise SystemExit(f"{OUT.name} is missing. Run without --check to build it.")
    data = json.loads(OUT.read_text("utf-8"))

    problems = []
    for key, sentence in data.get("explanation", {}).items():
        if _banned(sentence):
            problems.append(f"explanation {key}: {_banned(sentence)}")
    for key, questions in data.get("questions", {}).items():
        for question in questions:
            if not question.endswith("?"):
                problems.append(f"questions {key}: not a question -- {question!r}")
            if _banned(question):
                problems.append(f"questions {key}: {_banned(question)}")

    if problems:
        raise SystemExit("committed phrasings violate the rules:\n  "
                         + "\n  ".join(problems))
    log.info("%s: %d explanations, %d question sets, all within the rules",
             OUT.name, len(data.get("explanation", {})), len(data.get("questions", {})))
    return data


def dry_run():
    """Show exactly what would be sent -- no user data appears anywhere in it."""
    kind, key, prompt = JOBS[0]
    url, headers, body = build_request(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    )
    log.info("%d prompts would be sent. The first:\n", len(JOBS))
    log.info("POST %s", url.replace(headers.get("api-key", "x") or "x", "<key>"))
    log.info("%s", json.dumps(body, indent=2)[:1200])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.dry_run:
        dry_run()
        return
    if args.check:
        check()
        return

    log.info("generating %d phrasings", len(JOBS))
    output, rejected = generate()

    OUT.write_text(json.dumps(output, indent=2, sort_keys=True), "utf-8")
    log.info("")
    log.info("wrote %s: %d explanations, %d question sets",
             OUT.name, len(output["explanation"]), len(output["questions"]))
    if rejected:
        log.warning("rejected %d for breaking the rules:", len(rejected))
        for key, why in rejected:
            log.warning("   %s -- %s", key, why)


if __name__ == "__main__":
    main()
