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

@pytest.mark.parametrize("path", ["/sleep/", "/migraine/", "/heart_disease/"])
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
