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

# Comments quote the constructs they replaced, so assertions about what the
# stylesheet *does* have to read the rules, not the prose explaining them.
CSS_RULES = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)


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
    """It was 1,674 lines, most of it decoration. Guard the trim.

    Counts rules rather than raw lines. The guard is against decoration coming
    back, and a comment is not decoration -- this file explains the constructs
    it replaced, so prose grows every time a defect is fixed properly. Counting
    raw lines taxed the explanation and not the CSS: the print fixes added 48
    lines of rules and 38 of reasoning, and it was the reasoning that pushed it
    over 1,000.

    Dropping Bootstrap made this file *smaller*, which was not the expectation.
    It absorbed .btn-secondary -- the one component the CDN had been providing
    that this file had no rule for, so the "Back to Home" button had been left
    unstyled -- and lost twenty lines of --bs-* variable mappings that existed
    only to keep Bootstrap's palette in step with ours. There is no second
    palette now, so there is nothing to map.
    """
    rules = [line for line in CSS_RULES.splitlines() if line.strip()]
    assert len(rules) < 850


@pytest.mark.parametrize("selector", [
    ".skip-link", ":focus-visible", ".card", ".btn-primary", ".form-control",
    ".result-header", ".module-icon", ".form-steps-bar", ".toast",
    ".summary-entry", ".footer-disclaimer",
])
def test_core_components_are_defined(selector):
    assert selector in CSS, f"{selector} lost its styling"


@pytest.mark.parametrize("generated", [
    "gauge-chart-fill", "gauge-chart-percentage", "chart-track", "chart-tone",
    "toast-message", "toast-close", "ripple",
])
def test_javascript_generated_classes_are_still_styled(generated):
    """charts.js, toast.js and main.js build these at runtime, so a rename in
    the stylesheet silently unstyles them with nothing to catch it.

    `progress-circle-score`, `health-metric-fill` and `risk-level` used to be
    on this list. Nothing generated them -- their builders in charts.js had no
    caller and no data attribute in any template -- so the test was guarding
    the styling of components that could not appear. Both sides are gone now.
    """
    assert generated in CSS, f".{generated} is generated by JS but unstyled"


def test_every_class_charts_js_builds_is_styled():
    """The list above is hand-maintained, which is how it came to name three
    classes nothing produced. This reads the source instead, so a new class in
    charts.js cannot ship unstyled and an old one cannot linger unnoticed.
    """
    js = (STATIC / "js" / "charts.js").read_text(encoding="utf-8")
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    names = {n for group in re.findall(r'class="([a-z0-9 $-]+)"', js)
             for n in group.split() if n and "$" not in n}
    assert names, "no classes found; the template literal format must have changed"
    unstyled = sorted(n for n in names if f".{n}" not in CSS_RULES)
    assert not unstyled, f"charts.js builds these and nothing styles them: {unstyled}"


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


# --------------------------------------------------------------------------
# The app styles itself
#
# base.html used to load Bootstrap's CSS, its icon font and Inter from two
# CDNs, and style.css restyled .card and .btn on top of that -- but it never
# defined .container, .row or .col-md-6, which the templates use 145 times
# between them. So a blocked CDN did not degrade the page, it collapsed it into
# one unstyled column, and nothing here could see that: every test in this file
# passed with the CDN and passes without it.
#
# These are the tests that would have caught it.
# --------------------------------------------------------------------------

def _stylesheets():
    """Every rule the app serves: both CSS files and the per-page <style> blocks.

    Several pages carry their own <style> for classes only they use -- the
    nutrition traffic-light pills, the lifestyle score bars -- so a check that
    read only css/ would report a page's own styling as missing.
    """
    sheets = [p.read_text(encoding="utf-8") for p in (STATIC / "css").glob("*.css")]
    for path in TEMPLATES.rglob("*.html"):
        sheets += re.findall(r"<style[^>]*>(.*?)</style>",
                             path.read_text(encoding="utf-8"), re.S)
    return "\n".join(sheets)


def _classes_used():
    """``{class name: {templates using it}}``, ignoring Jinja expressions."""
    used = {}
    for path in TEMPLATES.rglob("*.html"):
        for attribute in re.findall(r'class="([^"]*)"',
                                    path.read_text(encoding="utf-8")):
            # `class="badge bg-{{ flag.urgency }}"` contributes "badge" only:
            # the rest is decided at render time and is covered by the tests
            # that exercise the routes.
            attribute = re.sub(r"\{[{%].*?[%}]\}", " ", attribute, flags=re.S)
            for name in attribute.split():
                if re.fullmatch(r"[a-z][a-z0-9]+(-[a-z0-9]+)*", name):
                    used.setdefault(name, set()).add(path.name)
    return used


def test_every_class_the_templates_use_is_defined_somewhere():
    """No class may depend on a stylesheet this app does not serve.

    This is the check that was missing. .container, .row, .col-md-6 and every
    spacing utility were coming from jsDelivr, so the layout of a health tool
    was contingent on a third party being reachable -- from a corporate proxy,
    from behind a national firewall, from a clinic having a bad morning.
    """
    defined = set(re.findall(r"\.(-?[A-Za-z_][A-Za-z0-9_-]*)", _stylesheets()))
    missing = {name: sorted(where)
               for name, where in _classes_used().items() if name not in defined}
    assert not missing, (
        "classes with no rule anywhere in this repository:\n  "
        + "\n  ".join(f"{name} -- used in {', '.join(where)}"
                      for name, where in sorted(missing.items()))
    )


def test_no_template_loads_anything_from_a_third_party():
    """Every byte a visitor's browser fetches comes from this origin.

    Beyond availability, this is the privacy claim: /privacy opens by saying
    nothing you enter is stored, and a health page that announces its visitor
    to Google Fonts and jsDelivr on every load undercuts that before a single
    question is answered.
    """
    offenders = []
    for path in TEMPLATES.rglob("*.html"):
        text = path.read_text(encoding="utf-8")
        # Comments explain what was removed and why; they are prose, not requests.
        text = re.sub(r"\{#.*?#\}|<!--.*?-->", "", text, flags=re.S)
        for match in re.findall(r'(?:src|href)="(https?://[^"]+)"', text):
            offenders.append(f"{path.name}: {match}")
    assert not offenders, (
        "templates fetching from another origin:\n  " + "\n  ".join(offenders)
    )


def test_the_icon_sprite_covers_every_icon_referenced():
    """A missing symbol is an invisible icon, not an error.

    The sprite is generated by tools/build_icon_sprite.py from whatever the
    templates, the JavaScript and app/ml/triage.py reference, so this fails
    when an icon has been added to a page and the sprite not rebuilt.
    """
    sprite = (TEMPLATES / "_icons.html").read_text(encoding="utf-8")
    available = set(re.findall(r'<symbol id="i-([a-z0-9-]+)"', sprite))

    referenced = set()
    for path in list(TEMPLATES.rglob("*.html")) + list((STATIC / "js").glob("*.js")):
        text = path.read_text(encoding="utf-8")
        referenced.update(re.findall(r'href="#i-([a-z0-9-]+)"', text))

    from app.ml.triage import CONCERNS
    referenced.update(concern.icon for concern in CONCERNS)

    assert not referenced - available, (
        f"referenced but not in the sprite: {sorted(referenced - available)}. "
        f"Run: python tools/build_icon_sprite.py"
    )
    # The other direction, so the sprite does not quietly accumulate weight it
    # is inlined into every page.
    assert not available - referenced, (
        f"in the sprite but unused: {sorted(available - referenced)}"
    )


def test_the_layout_layer_stayed_a_layer():
    """layout.css provides what the templates use, not a Bootstrap clone.

    The whole reason it is small enough to own is that it is a subset. If it
    ever approaches the thing it replaced, the trade stops being worth it.
    """
    layout = (STATIC / "css" / "layout.css").read_text(encoding="utf-8")
    assert len(layout) < 30_000, f"layout.css is {len(layout) / 1000:.1f} kB"

    # Read the rules, not the prose explaining them -- the same trap this
    # file's header warns about, which this test fell into first time out: the
    # comment describing what was replaced says ".card and .btn", and the
    # assertion matched its own explanation.
    rules = re.sub(r"/\*.*?\*/", "", layout, flags=re.S)
    for component in (".card", ".btn", ".alert", ".navbar"):
        assert not re.search(rf"^\s*\{re.escape(component)}[\s{{,:]", rules, re.M), (
            f"{component} belongs in style.css, not layout.css"
        )


def test_hidden_beats_bootstrap_display_utilities():
    """`.d-grid` sets `display: grid !important`, which outranks the user-agent
    `[hidden] { display: none }`. The submit block marked data-step-actions
    therefore stayed visible on step 1 of 4 -- a working form, showing the
    wrong control."""
    match = re.search(r"\[hidden\]\s*\{([^}]*)\}", CSS_RULES)
    assert match, "[hidden] is not styled at all"
    assert "display: none !important" in match.group(1)


def test_white_backgrounds_follow_the_theme():
    """`bg-white` left literal painted a glaring white band inside dark cards."""
    assert re.search(r"\.bg-white\s*\{[^}]*var\(--surface\)", CSS_RULES)


@pytest.mark.parametrize("selector,expected", [
    (r"\.form-label", "--ink"),
    (r"\.card-title", "--ink"),
])
def test_text_components_set_their_own_colour(selector, expected):
    match = re.search(selector + r"\s*\{([^}]*)\}", CSS_RULES)
    assert match and expected in match.group(1), f"{selector} inherits its colour"


def test_semantic_button_colours_are_not_used_for_navigation():
    """Colour carries meaning is the whole rule. The homepage had "Check Risk"
    in danger red and "Analyze Sleep" in info blue -- module links wearing
    status colours, which made the tile row read as an alert board."""
    index = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    for cls in ("btn-danger", "btn-warning", "btn-info", "btn-success"):
        assert cls not in index, f"{cls} used decoratively on the homepage"
