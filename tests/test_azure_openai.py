"""Azure OpenAI: what it may touch, and what happens when it can't be reached.

/privacy promises that assessment answers never leave this server. Adding a
third-party language model is exactly the change that could quietly make that
false, so the tests that matter most here are the ones asserting what is in the
outbound request body.

The split: the free-text sentence from /start is sent at runtime; the result
copy is generated offline and committed. Nothing from an assessment form is
ever transmitted.
"""

import json

import pytest

from app import azure_openai
from app.ml import phrasings, triage

SLEEP_FORM = {
    "snoring": "3", "gasping": "1", "sleepiness": "2", "insomnia_nights": "5",
    "insomnia_months": "1", "insomnia_impact": "1", "sleep_hours": "4.5",
}


@pytest.fixture()
def configured(monkeypatch):
    monkeypatch.setattr(azure_openai, "ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setattr(azure_openai, "API_KEY", "test-key")
    monkeypatch.setattr(azure_openai, "DEPLOYMENT", "gpt-test")


def _reply(content):
    """A minimal Azure chat-completions response."""
    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

    return Response()


# --------------------------------------------------------------------------
# off by default
# --------------------------------------------------------------------------

def test_absent_configuration_is_the_default():
    """The app must behave identically for anyone without credentials."""
    assert not azure_openai.is_configured()
    with pytest.raises(azure_openai.AzureUnavailable, match="not configured"):
        azure_openai.complete([{"role": "user", "content": "hi"}])


def test_routing_falls_back_to_keywords_when_unconfigured():
    matches, method = triage.route("I snore and I'm tired all day")
    assert method == "keywords"
    assert matches and matches[0][0].key == "sleep"


def test_healthz_reports_whether_it_is_on(client):
    payload = client.get("/healthz").get_json()
    assert "azure_openai" in payload
    assert payload["azure_openai"]["configured"] is False
    # Never leak the key, even to a probe endpoint.
    assert "api_key" not in json.dumps(payload).lower()


# --------------------------------------------------------------------------
# what leaves the server
# --------------------------------------------------------------------------

def test_only_the_typed_sentence_is_sent(configured, monkeypatch, client):
    """The single most important test in this file.

    Routing may see what the person typed into the box. It may not see
    anything they entered into an assessment.
    """
    sent = []

    def capture(url, headers=None, json=None, timeout=None):
        sent.append(json)
        return _reply('{"concerns": ["sleep"]}')

    monkeypatch.setattr(azure_openai.requests, "post", capture)

    # Run an assessment first. Nothing from it may reach the later call.
    client.post("/sleep/", data=SLEEP_FORM)

    typed = "I snore and wake up tired"
    client.post("/start", data={"concern": typed})

    assert len(sent) == 1, "an assessment route also called Azure"
    body = sent[0]

    # The claim is not "no scary words appear" -- the prompt legitimately names
    # every assessment, so words like BMI are in the static catalogue. The
    # claim is stronger: the body is *entirely determined* by fixed app text
    # plus the sentence typed. Anything user-specific would break this.
    catalogue = "\n".join(
        f"- {c.key}: {c.title} — {c.blurb}" for c in triage.CONCERNS
    )
    expected_user = f"Available tools:\n{catalogue}\n\nThe person wrote: {typed!r}"

    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]
    assert body["messages"][1]["content"] == expected_user
    assert typed in body["messages"][1]["content"]


def test_assessments_never_call_azure(configured, monkeypatch, client):
    """Result pages must not reach the network at all."""
    def explode(*args, **kwargs):
        raise AssertionError("an assessment route called Azure OpenAI")

    monkeypatch.setattr(azure_openai.requests, "post", explode)

    client.post("/sleep/", data=SLEEP_FORM)
    client.post("/health-score/", data={
        "Age": "35", "BMI": "22", "ExerciseFrequency": "5", "DietQuality": "8",
        "SleepHours": "8", "SmokingStatus": "0", "AlcoholConsumption": "1",
    })
    client.post("/heart_disease/", data={
        "high_bp": "1", "high_chol": "1", "chol_check": "1", "bmi": "34",
        "smoker": "1", "stroke": "0", "diabetes": "2", "phys_activity": "0",
        "alcohol": "0", "gen_health": "4", "ment_health": "15",
        "phys_health": "20", "diff_walk": "1", "sex": "1", "age": "68",
    })


def test_the_request_carries_the_key_in_a_header_not_the_body(configured):
    url, headers, body = azure_openai.build_request(
        [{"role": "user", "content": "hello"}]
    )
    assert headers["api-key"] == "test-key"
    assert "test-key" not in json.dumps(body)
    assert "test-key" not in url


# --------------------------------------------------------------------------
# routing behaviour
# --------------------------------------------------------------------------

def test_the_model_can_only_choose_a_real_destination(configured, monkeypatch):
    """A hallucinated key must not become a route."""
    monkeypatch.setattr(azure_openai.requests, "post",
                        lambda *a, **k: _reply('{"concerns": ["oncology", "sleep"]}'))
    matches, method = triage.route("I snore")
    assert method == "model"
    assert [c.key for c, _ in matches] == ["sleep"]


def test_a_wholly_invented_answer_falls_back_to_keywords(configured, monkeypatch):
    monkeypatch.setattr(azure_openai.requests, "post",
                        lambda *a, **k: _reply('{"concerns": ["oncology"]}'))
    matches, method = triage.route("I snore and I am tired")
    assert method == "keywords"
    assert matches and matches[0][0].key == "sleep"


@pytest.mark.parametrize("failure", [
    lambda *a, **k: (_ for _ in ()).throw(azure_openai.requests.Timeout()),
    lambda *a, **k: _reply("not json at all"),
    lambda *a, **k: _reply(None),
])
def test_every_failure_mode_falls_back(configured, monkeypatch, failure):
    """Azure being slow, wrong or filtered is ordinary, not an error page."""
    monkeypatch.setattr(azure_openai.requests, "post", failure)
    matches, method = triage.route("I snore and I am tired")
    assert method == "keywords"
    assert matches


def test_emergencies_are_detected_before_any_network_call(configured, monkeypatch):
    """Whether someone is told to call an ambulance must never depend on a
    third party being reachable."""
    def explode(*args, **kwargs):
        raise AssertionError("emergency detection reached the network")

    monkeypatch.setattr(azure_openai.requests, "post", explode)
    assert [s.key for s in triage.check_emergency("I have crushing chest pain")]


def test_the_emergency_stop_still_fires_with_azure_on(configured, monkeypatch, client):
    monkeypatch.setattr(azure_openai.requests, "post",
                        lambda *a, **k: _reply('{"concerns": ["heart"]}'))
    body = client.post(
        "/start", data={"concern": "I have crushing chest pain"}
    ).get_data(as_text=True)
    assert "Please get medical help now" in body
    assert "Start this assessment" not in body


def test_the_page_says_which_method_it_used(configured, monkeypatch, client):
    monkeypatch.setattr(azure_openai.requests, "post",
                        lambda *a, **k: _reply('{"concerns": ["sleep"]}'))
    body = client.post("/start", data={"concern": "I snore"}).get_data(as_text=True)
    assert "Worked out from what you wrote" in body
    assert "Matched on:" not in body  # there are no keywords to show


# --------------------------------------------------------------------------
# build-time phrasings
# --------------------------------------------------------------------------

def test_phrasings_are_optional(monkeypatch, tmp_path):
    """A missing file must leave every page working on its built-in copy."""
    monkeypatch.setattr(phrasings, "PATH", tmp_path / "absent.json")
    phrasings.reset()
    assert not phrasings.available()
    assert phrasings.explanation_for("BMI", "raised") is None
    assert phrasings.questions_for_band("heart disease risk", "high") == []
    phrasings.reset()


def test_phrasings_are_used_when_present(monkeypatch, tmp_path):
    path = tmp_path / "phrasings.json"
    path.write_text(json.dumps({
        "explanation": {"BMI|raised": "Carrying extra weight adds strain over time."},
        "questions": {"heart disease risk|high": ["Is a cholesterol test due?"]},
    }), encoding="utf-8")
    monkeypatch.setattr(phrasings, "PATH", path)
    phrasings.reset()

    assert phrasings.available()
    assert "extra weight" in phrasings.explanation_for("BMI", "raised")
    assert phrasings.questions_for_band("heart disease risk", "high")
    phrasings.reset()


def test_corrupt_phrasings_do_not_break_the_app(monkeypatch, tmp_path, client):
    path = tmp_path / "phrasings.json"
    path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(phrasings, "PATH", path)
    phrasings.reset()

    assert phrasings.explanation_for("BMI", "raised") is None
    assert client.post("/sleep/", data=SLEEP_FORM).status_code == 200
    phrasings.reset()


def test_generated_copy_is_screened_for_instructions():
    """Generated text goes in front of people deciding whether to seek care.

    A model ignoring "never instruct" is ordinary, so the generator rejects
    rather than commits.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "genphr",
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "ml_model" / "generate_phrasings.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._banned("You should stop taking your medication")
    assert module._banned("Don't worry about this result")
    assert module._banned("You have heart disease")
    assert not module._banned("Higher blood pressure is linked with greater risk.")


# --------------------------------------------------------------------------
# the privacy page must describe the state the app is actually in
# --------------------------------------------------------------------------

def test_privacy_page_is_silent_about_azure_when_it_is_off(client):
    body = client.get("/privacy").get_data(as_text=True)
    assert "Azure" not in body
    assert "Two things do leave this server" in " ".join(body.split())
    assert "Every model runs locally" in body


def test_privacy_page_declares_azure_when_it_is_on(configured, client):
    """Turning it on must change the page, or the page becomes a lie."""
    body = client.get("/privacy").get_data(as_text=True)
    flat = " ".join(body.split())

    assert "Azure OpenAI" in body
    assert "Three things do leave this server" in flat
    # And the absolute claim is downgraded to the accurate one.
    assert "Every model runs locally" not in body
    assert "never sees an answer" in flat


def test_the_start_page_declares_it_too(configured, client):
    body = client.get("/start").get_data(as_text=True)
    assert "language model" in body
    assert "Nothing from any assessment is" in " ".join(body.split())
