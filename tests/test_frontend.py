"""Front-end guarantees: assets resolve, pages work without JavaScript,
and the markup gives assistive technology something to work with.

Each of these covers a defect that was live: three favicon PNGs and an
og-image that never existed, a CSS rule that hid the whole page until a script
un-hid it, and native form validation suppressed in favour of a toast that read
the wrong element.
"""

import html
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tests.test_routes import FORMS

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


def test_no_third_party_stylesheet_hides_content_until_a_script_reveals_it():
    """The two tests above read style.css and main.js. Both passed the whole
    time the homepage rendered completely blank without JavaScript.

    AOS was loaded from unpkg. Its stylesheet sets opacity:0 on everything
    carrying `data-aos` and its script restores it on scroll, so all seventeen
    elements on the homepage -- the headline, the tagline and all six
    assessments -- were permanently invisible if the script did not run, and a
    CDN was a single point of failure for the landing page of a health tool.

    A test reading local files cannot evaluate a third party's CSS, so this
    guards the mechanism rather than the symptom: nothing in the markup may
    depend on a scroll-reveal library to become visible.
    """
    # Comments are stripped for the same reason _code_only exists: the note
    # explaining why AOS was removed necessarily names it.
    markup = "\n".join(p.read_text(encoding="utf-8")
                       for p in TEMPLATES.rglob("*.html"))
    markup = re.sub(r"\{#.*?#\}", "", markup, flags=re.S)
    markup = re.sub(r"<!--.*?-->", "", markup, flags=re.S).lower()
    assert "data-aos" not in markup, (
        "elements marked for AOS are invisible until its script runs"
    )
    for library in ("aos@", "aos.css", "aos.js", "wow.min", "scrollreveal"):
        assert library not in markup, (
            f"{library} hides content until it has loaded and run"
        )


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
# the homepage
# --------------------------------------------------------------------------

def test_the_homepage_offers_every_assessment(client):
    """It advertised six checks and, for anyone whose scroll never reached
    them, showed three.

    The six cards were also hard-coded here with their own titles and blurbs,
    beside a triage page generating its list from CONCERNS. Two descriptions of
    one set of assessments is one more than can be kept true, so the page now
    reads the same tuple the router does -- and this fails if one is added
    without appearing on the landing page.
    """
    from app.ml.triage import CONCERNS

    # Unescaped: autoescaping turns the apostrophe in "What's in the food I
    # eat" into &#39;, so a raw substring check misses it.
    body = html.unescape(client.get("/").get_data(as_text=True))
    missing = [c.key for c in CONCERNS if c.title not in body]
    assert not missing, f"assessments the homepage never mentions: {missing}"


def test_the_homepage_symptom_box_still_goes_through_the_emergency_check(client):
    """The box is the real field now, not a button leading to it.

    That is only safe while it posts to /start, where `check_emergency` runs
    before any routing. If it ever became a GET, or pointed at an assessment
    directly, someone typing "chest pain" on the landing page would be handed a
    questionnaire instead of a stop sign.
    """
    body = client.get("/").get_data(as_text=True)
    form = re.search(r'<form[^>]*action="/start"[^>]*>', body)
    assert form, "the homepage concern box no longer posts to /start"
    assert re.search(r'method="post"', form.group(0), re.I), (
        "the concern box must POST; a GET skips the emergency check"
    )

    urgent = client.post("/start", data={"concern": "crushing chest pain"})
    assert "Please get medical help now" in urgent.get_data(as_text=True)


# --------------------------------------------------------------------------
# accessibility
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", PAGES)
def test_landmarks_and_skip_link(client, path):
    """The skip link has to land somewhere that can hold focus.

    This test used to assert the link existed and `id="mainContent"` existed,
    and passed on every page while the link did nothing at all: <main> was not
    focusable, so following it moved the scroll position and left focus on the
    link. The next Tab went to the navbar brand -- back into the navigation the
    link exists to skip. Both halves being present was never the claim worth
    checking; that they connect is.
    """
    body = client.get(path).get_data(as_text=True)
    assert "<main" in body and "</main>" in body

    link = re.search(r'<a\s[^>]*class="skip-link"[^>]*>', body)
    assert link, "no skip link"
    href = re.search(r'href="#([^"]+)"', link.group(0))
    assert href, "the skip link does not point at a fragment"

    target = re.search(rf'<(\w+)\s[^>]*id="{re.escape(href.group(1))}"[^>]*>', body)
    assert target, f"the skip link points at #{href.group(1)}, which is not on the page"
    focusable = ("tabindex" in target.group(0)
                 or target.group(1) in ("a", "button", "input", "select", "textarea"))
    assert focusable, (
        f"<{target.group(1)} id={href.group(1)}> cannot receive focus, so the "
        "skip link only scrolls -- a keyboard user stays in the navigation"
    )


def test_nothing_autofocuses_past_the_skip_link():
    """`autofocus` starts focus inside the page, so the first Tab goes to
    whatever follows the focused field and the skip link is never reached.

    I put one on the /start box while building the conversational triage,
    which silently undid the skip-link fix two commits earlier. The static
    check above could not see it -- the markup was still correct, the focus
    order was not.
    """
    offenders = [p.name for p in TEMPLATES.rglob("*.html")
                 if "autofocus" in p.read_text(encoding="utf-8")]
    assert not offenders, (
        f"autofocus makes the skip link unreachable: {offenders}"
    )


def test_an_in_page_link_moves_focus_and_not_only_the_scroll():
    """main.js intercepts every `a[href^="#"]` for smooth scrolling.

    `preventDefault()` cancels the fragment navigation, and with it the focus
    move the browser would have performed -- which is what broke the skip link
    even once <main> was made focusable. Anything that takes over an in-page
    link has to do the whole job, not the visible half of it.
    """
    js = _code_only(STATIC / "js" / "main.js")
    start = js.index('a[href^="#"]')
    open_brace = js.index("{", start)
    depth, end = 0, None
    for i in range(open_brace, len(js)):
        depth += (js[i] == "{") - (js[i] == "}")
        if depth == 0:
            end = i
            break
    assert end, "could not find the end of the smooth-scroll handler"
    handler = js[start:end]
    assert "preventDefault" in handler
    assert ".focus(" in handler, (
        "the smooth-scroll handler scrolls sighted users to the target and "
        "leaves keyboard focus behind"
    )


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


# --------------------------------------------------------------------------
# the printed visit summary
# --------------------------------------------------------------------------

def _print_block():
    css = _code_only(STATIC / "css" / "style.css")
    start = css.index("@media print")
    depth, i = 0, start
    while i < len(css):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                return css[start:i + 1]
        i += 1
    raise AssertionError("unterminated @media print block")


def test_print_stylesheet_exists_for_the_visit_summary():
    block = _print_block()
    # The app furniture must not end up on a page handed to a doctor.
    for selector in ("nav", "footer", ".summary-toolbar"):
        assert selector in block


def test_printing_does_not_inherit_the_dark_theme():
    """The visit summary is the artifact this whole feature exists for, and it
    printed pale grey on white for anyone whose OS was in dark mode.

    `@media print` set `color` on `body` and a white background on `.card`.
    Everything else -- headings, `.text-muted`, every question in the list --
    reads its colour from the design tokens, which `prefers-color-scheme: dark`
    had already switched to near-white. In light mode it printed correctly,
    which is why it survived being looked at.

    Asserting the tokens are reset covers every descendant at once.
    """
    block = _print_block()
    assert ":root" in block, "print must reset the theme tokens, not one element"
    for token in ("--ink:", "--ink-muted:", "--surface:", "--page:"):
        assert token in block, f"{token} still inherited from the screen theme"

    # And nothing may be left resolving to a light-on-light value.
    for dark in ("#e8ecf3", "#a3adbf", "#12161f", "#1b2130"):
        assert dark not in block


def test_the_answers_behind_a_result_reach_the_printed_page():
    """"What I entered" is a collapsed <details>, and a closed <details> does
    not print -- so the inputs a doctor is most likely to question were absent.
    CSS cannot open one, so the print event has to.
    """
    js = _code_only(STATIC / "js" / "summary.js")
    assert "beforeprint" in js and "afterprint" in js
    assert ".summary-inputs" in js


def test_bootstrap_column_widths_do_not_squeeze_the_printed_labels():
    """The <dl> carries `.row` and each <dt> `.col-6 .col-sm-4`. Forcing grid
    on the parent left those percentages applying to grid items, so "high blood
    pressure" printed as three stacked lines inside a 117pt label column.
    """
    block = _print_block()
    assert "width: auto !important" in block


def test_the_print_block_outranks_the_dark_theme():
    """Both blocks open with `:root`, so source order decides which palette
    wins -- and both media queries match when you print from a machine whose
    OS is in dark mode.

    `@media print` used to sit at section 9 and dark mode at section 10, so the
    dark tokens won and the print reset was inert for exactly the readers it
    was written for. The test above asserts the reset exists; it passed the
    whole time, because it read the source rather than the cascade. Rendering
    the pages showed `--ink` still resolving to #e8ecf3 under print emulation.

    Order is the fix, so order is what this guards.
    """
    css = _code_only(STATIC / "css" / "style.css")
    assert css.index("@media print") > css.index("@media (prefers-color-scheme: dark)"), (
        "@media print must come after the dark-mode block or its :root reset "
        "loses on source order"
    )


def test_every_selector_in_the_print_block_is_one_the_app_uses():
    """The previous print block styled `.summary-question` and
    `.summary-questions li`. Neither has ever existed -- the questions render as
    `li` inside `#summaryQuestions` -- so the rule meant to make them legible
    matched nothing, and the seven questions a patient takes to a doctor kept
    printing in near-white.

    A print rule cannot be looked at in the normal course of using the app, so
    a selector typo there is invisible until someone prints. This is the check
    that would have caught it.
    """
    block = _print_block()
    corpus = "\n".join(
        p.read_text(encoding="utf-8")
        for p in list(TEMPLATES.rglob("*.html")) + list((STATIC / "js").rglob("*.js"))
    )
    css = _code_only(STATIC / "css" / "style.css")
    rest = css.replace(block, "")

    dead = []
    for name in sorted(set(re.findall(r"[.#][A-Za-z][\w-]*", block))):
        bare = re.escape(name[1:])
        if re.search(rf"\b{bare}\b", corpus) or re.search(rf"[.#]{bare}\b", rest):
            continue
        dead.append(name)
    assert not dead, f"print rules for selectors nothing uses: {dead}"


def test_white_text_on_a_colour_fill_is_given_ink_for_print():
    """A printer leaves background graphics off by default, so a white number
    on a `bg-danger` pill comes out as blank paper.

    Every result page put its headline figure in exactly that: "63.58%
    estimated risk" on heart, "97.5% estimated risk" and the whole accuracy
    note on migraine, the health-score rating, the calculator's BMI band. All
    of them printed as nothing at all, in light mode as well as dark, and the
    page still looked complete because the surrounding prose printed fine.
    """
    block = _print_block()
    for selector in (".text-white", ".badge", ".probability-badge",
                     ".bg-danger", ".bg-success", ".result-header"):
        assert selector in block, f"{selector} would print white on white"
    assert "background: none !important" in block


def test_a_result_card_may_break_across_pages():
    """`.card` used to carry `break-inside: avoid`, which suits a summary entry
    and not a result card taller than a sheet of A4. The browser cannot honour
    it, so it pushed the whole card to page 2 -- and the heart result printed a
    first page that was blank below the title.
    """
    block = _print_block()
    avoid = [seg for seg in block.split("}") if "break-inside: avoid" in seg]
    assert avoid, "nothing is protected from splitting across pages"
    for seg in avoid:
        selectors = seg.split("{")[0]
        assert not re.search(r"(^|,)\s*\.card\s*(,|$)", selectors), (
            "a result card is taller than a page; break-inside: avoid on .card "
            "blanks the page before it"
        )


# --------------------------------------------------------------------------
# what a screen reader is handed
# --------------------------------------------------------------------------

class _Structure(HTMLParser):
    """Collect headings, labels and controls, skipping hidden subtrees.

    Content inside `d-none` is not in the accessibility tree, and /summary
    carries a second <h1> in a print-only header that is `d-none` on screen --
    counting raw tags would call that a duplicate heading it is not.
    """

    SELF_CLOSING = {"input", "img", "br", "hr", "meta", "link"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.headings = []          # (level, text)
        self.labels = []            # for= values
        self.controls = []          # (tag, name, id, has_aria_label)
        self._stack = []
        self._hidden_depth = 0
        self._heading = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        hidden = ("d-none" in (a.get("class") or "").split()
                  or "hidden" in a
                  or a.get("aria-hidden") == "true")
        if tag not in self.SELF_CLOSING:
            self._stack.append((tag, hidden))
            if hidden:
                self._hidden_depth += 1
        if self._hidden_depth:
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._heading = [int(tag[1]), ""]
        elif tag == "label" and a.get("for"):
            self.labels.append(a["for"])
        elif tag in ("input", "select", "textarea"):
            if a.get("type") not in ("hidden", "submit", "button"):
                self.controls.append((tag, a.get("name"), a.get("id"),
                                      bool(a.get("aria-label")
                                           or a.get("aria-labelledby"))))

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self._heading:
            self.headings.append(tuple(self._heading))
            self._heading = None
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                if self._stack[i][1]:
                    self._hidden_depth -= 1
                del self._stack[i:]
                break

    def handle_data(self, data):
        if self._heading is not None and not self._hidden_depth:
            self._heading[1] += data


def _structure(client, path, data=None):
    response = client.post(path, data=data) if data else client.get(path)
    parser = _Structure()
    parser.feed(response.get_data(as_text=True))
    return parser


RESULTS = sorted(FORMS)


@pytest.mark.parametrize("path", PAGES)
def test_every_form_field_is_labelled(client, path):
    """`<label for="age">` against `<input name="age">` with no id at all.

    Four of the five forms did this -- heart, migraine, the health score and
    the calculator -- 41 fields between them. The label is right there in the
    source and right there on the screen, and the association a sighted reader
    infers from the layout does not exist for anyone who cannot see it: the
    accessibility tree gets an unnamed control, and a screen reader announces
    "edit, blank" forty-one times. The sleep form, rebuilt later, had ids
    throughout, which is why this was invisible to anyone spot-checking one
    page.
    """
    s = _structure(client, path)
    ids = {c[2] for c in s.controls if c[2]}
    unlabelled = [
        name or "(unnamed)" for tag, name, cid, aria in s.controls
        if not aria and (cid is None or cid not in s.labels)
    ]
    assert not unlabelled, f"{path}: fields with no usable label: {unlabelled}"

    dangling = [f for f in s.labels if f not in ids]
    assert not dangling, f"{path}: labels pointing at no control: {dangling}"


@pytest.mark.parametrize("path", PAGES)
def test_every_page_has_exactly_one_h1(client, path):
    """Every form and result page led with an <h2>, because <h1> looked too
    big -- so the pages a screen-reader user is most likely to navigate by
    heading had no level-one heading to navigate to.
    """
    s = _structure(client, path)
    ones = [text.strip() for level, text in s.headings if level == 1]
    assert len(ones) == 1, f"{path}: {len(ones)} visible <h1>: {ones}"


@pytest.mark.parametrize("path", PAGES)
def test_heading_levels_do_not_skip(client, path):
    """Heading level is the document outline. Picking a tag for its size --
    <h5> for a section title because it looked right -- turned that outline
    into h2, h5, h5, h5 on every form.
    """
    s = _structure(client, path)
    levels = [lvl for lvl, _ in s.headings]
    skips = [(levels[i - 1], levels[i], s.headings[i][1].strip()[:40])
             for i in range(1, len(levels)) if levels[i] - levels[i - 1] > 1]
    assert not skips, f"{path}: heading level jumps: {skips}"


@pytest.mark.parametrize("path", RESULTS)
def test_result_pages_are_structured_too(client, path):
    """The result page is the one a screen-reader user actually needs to read
    through, and it was the least structured: no <h1>, and jumps to <h6>.
    """
    s = _structure(client, path, data=FORMS[path])
    ones = [t.strip() for lvl, t in s.headings if lvl == 1]
    assert len(ones) == 1, f"{path}: {len(ones)} <h1> on the result: {ones}"
    levels = [lvl for lvl, _ in s.headings]
    skips = [(levels[i - 1], levels[i]) for i in range(1, len(levels))
             if levels[i] - levels[i - 1] > 1]
    assert not skips, f"{path}: heading level jumps on the result: {skips}"


def test_a_toast_is_announced_and_can_be_dismissed_by_name():
    """The toast container had no live region, so a message appended after
    load was never spoken -- including the one every heart result fires. Its
    close button was a tabbable control whose only content was an icon glyph,
    so it reached the accessibility tree unnamed.
    """
    js = _code_only(STATIC / "js" / "toast.js")
    assert "aria-live" in js, "toast messages are never announced"
    assert "'role', 'status'" in js or 'role", "status' in js
    assert "aria-label" in js, "the dismiss button has no accessible name"
    assert 'type="button"' in js
    # The icon repeats the message; it must not be read out as well.
    assert js.count('aria-hidden="true"') >= 4


def test_heading_size_classes_carry_the_same_typography_as_the_tags():
    """Fixing the outline means writing `<h2 class="h5">`, which only looks
    identical if `.h5` styles the same as `h5`. Bootstrap's `.h1`-`.h6` set
    font-weight and line-height as well as size, and a class beats an element
    selector -- so without these the app's headings silently fell back to
    Bootstrap's 500 weight and 1.2 line-height wherever a size class was used.
    """
    css = _code_only(STATIC / "css" / "style.css")
    # Anchored on the selector, not on a declaration inside it: the first
    # version of this matched `font-weight: 700` and broke the moment the
    # headings were restyled to 600, which is a change it should not have an
    # opinion about.
    block = re.search(r"([^}]*\bh1\s*,[^}]*?)\{", css)
    assert block, "the shared heading rule moved; check this test still applies"
    selectors = block.group(1)
    for name in (".h1", ".h2", ".h3", ".h4", ".h5", ".h6"):
        assert name in selectors, (
            f"{name} does not inherit the app's heading typography, so "
            f'<h2 class="{name[1:]}"> renders as Bootstrap rather than as this app'
        )


def test_printing_a_result_leaves_out_the_buttons():
    """On paper a button is an instruction to go back to a screen. "Take
    Another Assessment", "Back to Home" and the "Add to visit summary" card are
    the app talking to itself, and they were printing on every result page.
    """
    block = _print_block()
    hidden = [seg for seg in block.split("}") if "display: none" in seg]
    selectors = " ".join(seg.split("{")[0] for seg in hidden)
    assert ".btn" in selectors
    assert "[data-summary-card]" in selectors
