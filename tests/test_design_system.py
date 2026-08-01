"""The visual system holds together and stays readable.

Half the app had been rewritten in a plainer style while the other half kept
emoji headings and a gradient on every card, so it read as two products. These
tests are what keeps the two halves from drifting apart again.

The contrast checks matter more than the tidiness ones. This is a health tool
whose whole argument is that colour carries meaning -- red means seek care --
so the colours have to be legible to the people reading them.
"""

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
TEMPLATES = Path(__file__).resolve().parent.parent / "app" / "templates"
CSS = (STATIC / "css" / "style.css").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# contrast
# --------------------------------------------------------------------------

def _luminance(hex_colour):
    hex_colour = hex_colour.lstrip("#")
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a, b):
    la, lb = _luminance(a), _luminance(b)
    high, low = max(la, lb), min(la, lb)
    return (high + 0.05) / (low + 0.05)


def token(name, dark=False):
    """Read a custom property out of the stylesheet.

    Deliberately reads the real file rather than a copy of the values, so a
    palette change is caught here rather than in a screenshot months later.
    """
    blocks = CSS.split("@media (prefers-color-scheme: dark)")
    haystack = blocks[1] if dark and len(blocks) > 1 else blocks[0]
    match = re.search(rf"{re.escape(name)}:\s*(#[0-9a-fA-F]{{6}})", haystack)
    assert match, f"{name} not found in the stylesheet"
    return match.group(1)


LIGHT_ON_SURFACE = ["--ink", "--ink-muted", "--ink-faint", "--danger",
                    "--warning", "--success", "--info", "--brand"]


@pytest.mark.parametrize("name", LIGHT_ON_SURFACE)
def test_text_tokens_are_readable_on_the_surface(name):
    assert contrast(token(name), token("--surface")) >= 4.5, name


@pytest.mark.parametrize("name", ["--danger", "--warning", "--success", "--info", "--brand"])
def test_white_is_readable_on_the_solid_fills(name):
    """These fill the result headers and primary buttons."""
    assert contrast("#ffffff", token(name)) >= 4.5, name


@pytest.mark.parametrize("name", ["--ink", "--ink-muted", "--danger", "--warning",
                                  "--success", "--info", "--brand"])
def test_dark_mode_tokens_are_readable(name):
    assert contrast(token(name, dark=True), token("--surface", dark=True)) >= 4.5, name


# --------------------------------------------------------------------------
# consistency
# --------------------------------------------------------------------------

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⬀-⯿]")


def test_no_emoji_are_used_as_interface_elements():
    """Emoji render differently per platform and are read aloud oddly.

    result_migraine.html alone had 48 of them, including tick marks used as
    list bullets, which a screen reader announces as "white heavy check mark".
    """
    offenders = {
        path.name: EMOJI.findall(path.read_text(encoding="utf-8"))
        for path in TEMPLATES.rglob("*.html")
    }
    offenders = {name: found for name, found in offenders.items() if found}
    assert not offenders, f"emoji still in templates: {offenders}"


def test_gradients_are_confined_to_the_brand():
    """A gradient on every card meant none of them signalled anything.

    The hero keeps one -- it is identity, not information.
    """
    gradients = re.findall(r"linear-gradient\([^)]*\)", CSS)
    assert len(gradients) <= 1, f"{len(gradients)} gradients in the stylesheet"

    for path in TEMPLATES.rglob("*.html"):
        text = path.read_text(encoding="utf-8")
        assert "linear-gradient" not in text, f"{path.name} styles a gradient inline"


def test_semantic_colours_are_not_hardcoded_in_templates():
    """Bootstrap's raw hexes reappearing inline is how the system rots."""
    stray = {}
    for path in TEMPLATES.rglob("*.html"):
        found = re.findall(r"#(?:dc3545|28a745|198754|ffc107|667eea|764ba2|0d6efd)",
                           path.read_text(encoding="utf-8"))
        if found:
            stray[path.name] = sorted(set(found))
    assert not stray, f"hardcoded palette colours: {stray}"


def test_the_stylesheet_stayed_small():
    """It was 1,674 lines, most of it decoration. Guard the trim."""
    assert len(CSS.splitlines()) < 1000


@pytest.mark.parametrize("selector", [
    ".skip-link", ":focus-visible", ".card", ".btn-primary", ".form-control",
    ".result-header", ".module-icon", ".form-steps-bar", ".toast",
    ".summary-entry", ".footer-disclaimer",
])
def test_core_components_are_defined(selector):
    assert selector in CSS, f"{selector} lost its styling"


@pytest.mark.parametrize("generated", [
    "gauge-chart-fill", "progress-circle-score", "health-metric-fill",
    "toast-message", "toast-close", "risk-level", "ripple",
])
def test_javascript_generated_classes_are_still_styled(generated):
    """charts.js, toast.js and main.js build these at runtime, so a rename in
    the stylesheet silently unstyles them with nothing to catch it."""
    assert generated in CSS, f".{generated} is generated by JS but unstyled"


def test_icons_used_in_templates_are_marked_decorative():
    """A screen reader announcing every icon name is noise."""
    missing = []
    for path in TEMPLATES.rglob("*.html"):
        for tag in re.findall(r"<i class=\"bi [^\"]*\"[^>]*>", path.read_text(encoding="utf-8")):
            if "aria-hidden" not in tag and "aria-label" not in tag:
                missing.append(f"{path.name}: {tag[:60]}")
    # A handful of pre-existing decorative icons are inside labelled buttons;
    # anything above a small tail means the convention has been dropped.
    assert len(missing) <= 12, "icons without aria-hidden:\n  " + "\n  ".join(missing)
