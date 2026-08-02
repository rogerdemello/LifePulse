"""Front-end guarantees: assets resolve, pages work without JavaScript,
and the markup gives assistive technology something to work with.

Each of these covers a defect that was live: three favicon PNGs and an
og-image that never existed, a CSS rule that hid the whole page until a script
un-hid it, and native form validation suppressed in favour of a toast that read
the wrong element.
"""

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
TEMPLATES = Path(__file__).resolve().parent.parent / "app" / "templates"

PAGES = [
    "/", "/privacy", "/summary", "/health/", "/sleep/",
    "/heart_disease/", "/migraine/", "/health-score/", "/nutrition/",
]


def _code_only(path):
    """Strip comments before asserting on source.

    Several of these tests check that a specific broken construct is gone. The
    code that replaced it carries a comment explaining what it replaced and why
    -- which naturally quotes the construct. Matching raw text would fail on the
    documentation of the fix.
    """
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)          # /* block */
    if path.suffix == ".js":
        text = re.sub(r"^\s*//.*$", "", text, flags=re.M)      # // line
    return text


# --------------------------------------------------------------------------
# assets
# --------------------------------------------------------------------------

def test_every_referenced_static_file_exists():
    """Four files were referenced and missing, 404ing on every page load.

    The og-image one was invisible in the browser but broke the preview card on
    every shared link.
    """
    pattern = re.compile(r"filename=['\"]([^'\"]+)['\"]")
    missing = []
    for template in TEMPLATES.rglob("*.html"):
        for reference in pattern.findall(template.read_text(encoding="utf-8")):
            if not (STATIC / reference).exists():
                missing.append(f"{template.name} -> {reference}")
    assert not missing, "referenced but absent:\n  " + "\n  ".join(missing)


def test_favicon_is_inlined_not_a_missing_file(client):
    body = client.get("/").get_data(as_text=True)
    assert 'rel="icon"' in body
    assert "favicon-32x32" not in body
    assert "favicon-16x16" not in body


# --------------------------------------------------------------------------
# works without JavaScript
# --------------------------------------------------------------------------

def test_no_css_rule_hides_the_page_until_javascript_runs():
    """`body { opacity: 0 }` left the site permanently blank if a script failed."""
    css = _code_only(STATIC / "css" / "style.css")
    body_rule = re.search(r"\bbody\s*\{(.*?)\}", css, re.S)
    assert body_rule
    assert "opacity: 0" not in body_rule.group(1)


def test_no_script_hides_the_page_until_javascript_runs():
    js = _code_only(STATIC / "js" / "main.js")
    assert "document.body.style.opacity" not in js


@pytest.mark.parametrize("path", PAGES)
def test_content_is_present_in_the_served_html(client, path):
    """Content must be in the markup, not assembled by a script.

    The one exception is /summary, which is a deliberate client-side shell --
    the saved results live in the browser and never reach the server.
    """
    body = client.get(path).get_data(as_text=True)
    assert "<main" in body
    assert "screening tool, not a diagnosis" in body
    if path != "/summary":
        assert len(re.sub(r"<[^>]+>", "", body).strip()) > 500


def test_native_form_validation_is_not_suppressed():
    """The `invalid` handler used to preventDefault and show its own toast.

    That killed the browser's message and its focus jump, and the toast text
    came from previousElementSibling -- for the blood-pressure input group,
    the "/" separator. It announced "/ is required".
    """
    js = _code_only(STATIC / "js" / "main.js")
    invalid_handler = re.search(
        r"addEventListener\('invalid'.*?\}\);", js, re.S
    )
    assert invalid_handler
    assert "preventDefault" not in invalid_handler.group(0)
    assert "previousElementSibling" not in js


# --------------------------------------------------------------------------
# accessibility
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", PAGES)
def test_landmarks_and_skip_link(client, path):
    body = client.get(path).get_data(as_text=True)
    assert 'class="skip-link"' in body
    assert 'id="mainContent"' in body
    assert "<main" in body and "</main>" in body


@pytest.mark.parametrize("path,expected", [
    ("/sleep/", "/sleep"),
    ("/migraine/", "/migraine"),
    ("/nutrition/", "/nutrition"),
    ("/", "/"),
])
def test_current_page_is_marked_in_the_nav(client, path, expected):
    body = client.get(path).get_data(as_text=True)
    assert 'aria-current="page"' in body
    marked = re.search(r'href="([^"]+)"[^>]*aria-current="page"', body) or \
             re.search(r'aria-current="page"[^>]*href="([^"]+)"', body)
    assert marked, "aria-current is not attached to a link"


def test_reduced_motion_is_honoured():
    """This app has a page about migraine triggers; animation is one."""
    css = (STATIC / "css" / "style.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in css


def test_dark_mode_is_supported():
    css = (STATIC / "css" / "style.css").read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in css


def test_keyboard_focus_is_visible():
    css = (STATIC / "css" / "style.css").read_text(encoding="utf-8")
    assert ":focus-visible" in css


def test_the_loading_overlay_cannot_strand_a_back_navigation():
    """bfcache restores the page with the overlay still covering it."""
    js = (STATIC / "js" / "main.js").read_text(encoding="utf-8")
    assert "pageshow" in js


# --------------------------------------------------------------------------
# honest limitations in the UI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/migraine/", "/heart_disease/"])
def test_binary_sex_field_explains_itself(client, path):
    """The models only know two values. Say so rather than implying otherwise."""
    body = client.get(path).get_data(as_text=True)
    assert "only recorded two values" in body


def test_print_stylesheet_exists_for_the_visit_summary():
    css = (STATIC / "css" / "style.css").read_text(encoding="utf-8")
    assert "@media print" in css
    print_block = css[css.index("@media print"):]
    # The app furniture must not end up on a page handed to a doctor.
    for selector in ("nav", "footer", ".summary-toolbar"):
        assert selector in print_block


# --------------------------------------------------------------------------
# multi-step forms
# --------------------------------------------------------------------------

STEPPED = ["/heart_disease/", "/sleep/"]


@pytest.mark.parametrize("path", STEPPED)
def test_stepped_forms_declare_their_steps(client, path):
    body = client.get(path).get_data(as_text=True)
    assert "data-steps" in body
    assert body.count('data-step="') >= 2
    assert "data-step-actions" in body, "no submit block for the last step"


@pytest.mark.parametrize("path", STEPPED)
def test_every_field_is_in_the_markup_not_built_by_script(client, path):
    """The steps are an enhancement. With JavaScript off the whole form must
    still be present and submittable, which is why this hides sections from
    script rather than from CSS."""
    body = client.get(path).get_data(as_text=True)
    form = body[body.index("<form"):body.index("</form>")]
    inputs = re.findall(r'<(?:input|select)\b[^>]*name="([^"]+)"', form)
    assert len(inputs) >= 7, f"{path} only exposed {inputs}"


def test_steps_are_not_hidden_by_css():
    """A CSS rule hiding steps would break the form for anyone without JS.

    This is the same failure as the old `body { opacity: 0 }`: styling that
    assumes a script will arrive to undo it.
    """
    css = _code_only(STATIC / "css" / "style.css")
    assert "[data-step]" not in css or "display: none" not in css


def test_the_step_script_only_hides_what_it_grouped():
    js = _code_only(STATIC / "js" / "steps.js")
    # Bails out rather than hiding anything when there is nothing to group.
    assert "if (steps.length < 2) return;" in js
    # Native validation is used rather than replaced.
    assert "reportValidity" in js
    # And the change of step is announced.
    assert "aria-live" in js


# --------------------------------------------------------------------------
# the privacy page has to describe what the app actually does
# --------------------------------------------------------------------------

def test_privacy_page_discloses_browser_storage(client):
    """The visit summary writes results -- and the answers behind them -- to
    localStorage. Until this test existed, the privacy page still said
    "Because nothing is saved, you can't come back later and find a result",
    which the feature had made false.

    A privacy page that is out of date is worse than none: it is the page
    someone reads *instead of* checking.
    """
    body = " ".join(client.get("/privacy").get_data(as_text=True).split())

    # It must name the mechanism, say where it lives, and say what removes it.
    assert "local storage" in body.lower()
    assert "never uploaded" in body
    assert "Clear" in body

    # And it must not claim the absolute that the summary feature broke.
    assert "Because nothing is saved" not in body
    assert "writes nothing to disk" not in body


def test_the_storage_claim_is_scoped_to_the_server(client):
    """"Nothing you enter is stored" appears in the footer of every page.

    It is only true of the server now, and an unqualified version reads as a
    promise about the device -- which is where the summary actually lives.
    """
    for path in ("/", "/privacy", "/summary", "/nutrition/"):
        body = client.get(path).get_data(as_text=True)
        assert "Nothing you enter is stored on our server" in body, path


def test_summary_page_says_it_needs_javascript(client):
    """Every other page renders server-side; this one cannot."""
    body = client.get("/summary").get_data(as_text=True)
    assert "<noscript>" in body
    assert "needs JavaScript" in body


def test_the_footer_is_pinned_on_short_pages():
    """A footer band ending a third of the way up the window, with page
    background below it, reads as content that failed to load.
    """
    css = _code_only(STATIC / "css" / "style.css")
    assert "min-height: 100vh" in css
    assert "body > main" in css
