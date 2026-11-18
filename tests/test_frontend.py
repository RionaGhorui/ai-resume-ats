"""Headless Streamlit + stylesheet tests. No backend, no LLM, no network."""
import copy
import pathlib
import re

import pytest
from streamlit.testing.v1 import AppTest

from backend import analyzer
from tests.conftest import SAMPLE_TEXT

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "frontend" / "app.py"
CSS = ROOT / "frontend" / "styles.css"
CONFIG = ROOT / ".streamlit" / "config.toml"


@pytest.fixture(autouse=True)
def _never_use_a_real_key(monkeypatch):
    """A real GROQ_API_KEY in your .env must never leak into tests (it would make real API calls)."""
    monkeypatch.setattr("backend.config.GROQ_API_KEY", "")


@pytest.fixture
def result(monkeypatch, parsed_resume, parsed_jd):
    monkeypatch.setattr(analyzer, "parse_resume", lambda t: parsed_resume)
    monkeypatch.setattr(analyzer, "parse_job_description", lambda t: parsed_jd)
    monkeypatch.setattr(analyzer, "generate_suggestions", lambda *a, **k: [], raising=False)
    return analyzer.analyze(SAMPLE_TEXT, "jd").model_dump()


def run_app(session=None, timeout=30):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    for key, value in (session or {}).items():
        at.session_state[key] = value
    return at.run()


def markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def app_helpers() -> dict:
    """Load only the helper functions (everything before show()) without running the page."""
    ns: dict = {}
    exec(compile(APP.read_text().split("def show(")[0], "app_helpers", "exec"), ns)
    return ns


# ------------------------------------------------------------------ page
def test_page_loads_with_styles_and_hero():
    at = run_app()
    md = markdown(at)
    assert not at.exception
    assert "Check your resume the way an ATS reads it" in md
    assert "<style>" in md and "--pass" in md                      # styles.css was injected
    assert 'class="report' not in md                               # no result yet


def test_page_renders_a_result(result):
    at = run_app({"result": result})
    md = markdown(at)
    assert not at.exception
    assert re.search(r'class="report tone-(good|fair|low)"', md)
    assert f'aria-label="Score {result["ats_score"]:.0f} out of 100"' in md
    assert [t.label for t in at.tabs] == ["Fixes", "Skills", "Job match", "Download"]
    assert len(at.expander) == len(result["detailed_feedback"])


def test_score_tone_follows_the_score(result):
    h = app_helpers()
    assert 'style="--p:81.5"' in h["report_html"]({**result, "ats_score": 81.5})   # the inline variable the CSS relies on
    for score, tone in [(92, "good"), (70, "good"), (69.9, "fair"), (55, "fair"), (54.9, "low"), (10, "low")]:
        assert f"report tone-{tone}" in h["report_html"]({**result, "ats_score": score})


# ------------------------------------------------------------------ safety of the custom HTML
EVIL = '<img src=x onerror="alert(1)"><script>alert(2)</script> $x$'


def test_llm_text_is_escaped_everywhere(result):
    bad = copy.deepcopy(result)
    bad["skill_validation"]["unvalidated"] = [EVIL]
    bad["skill_validation"]["validated"] = [{"skill": EVIL, "evidence": [EVIL]}]
    bad["strengths"] = [EVIL]
    bad["suggestions"] = [{"target": EVIL, "advice": EVIL, "example_bullet": "ok"}]
    bad["interpretation"] = f"{EVIL}: {EVIL}"
    bad["jd_match"].update(matched_keywords=[EVIL], missing_keywords=[EVIL], skills_gap=[EVIL],
                           requirement_matches=[{"requirement": EVIL, "source": EVIL, "score": 0.5}])
    md = markdown(run_app({"result": bad}))
    assert "<img" not in md and "<script" not in md.split("</style>", 1)[1]
    assert "&lt;img" in md and "&#36;x&#36;" in md                  # escaped, and $ cannot start LaTeX


def test_custom_html_blocks_have_no_blank_lines(result):
    """A blank line ends an HTML block in Markdown, and the rest would be shown as raw text."""
    md_values = [m.value for m in run_app({"result": result}).markdown]
    blocks = [v for v in md_values if 'class="' in v and "<style>" not in v]
    assert len(blocks) >= 4
    assert all("\n\n" not in v and not re.search(r"\n {4}", v) for v in blocks)


# ------------------------------------------------------------------ stylesheet and theme
def test_stylesheet_is_well_formed():
    css = CSS.read_text()
    assert css.lstrip().startswith("@import")                      # @import must come first or it is ignored
    assert css.count("{") == css.count("}")
    defined = set(re.findall(r"(--[\w-]+)\s*:", css))
    used = set(re.findall(r"var\((--[\w-]+)", css))
    inline = {"--p"}                                               # the ring's fill %, set per ring by report_html()
    assert used - inline <= defined, f"undefined CSS variables: {used - inline - defined}"
    assert "prefers-reduced-motion" in css and ":focus-visible" in css and "max-width: 760px" in css


def test_theme_config_matches_the_stylesheet_and_the_api_limit():
    tomllib = pytest.importorskip("tomllib")                       # Python 3.11+
    cfg = tomllib.loads(CONFIG.read_text())
    css = CSS.read_text()
    var = lambda name: re.search(rf"--{name}:\s*(#[0-9A-Fa-f]{{6}})", css).group(1).lower()
    theme = {k: v.lower() for k, v in cfg["theme"].items() if str(v).startswith("#")}
    assert theme["primaryColor"] == var("pass")
    assert theme["backgroundColor"] == var("desk")
    assert theme["textColor"] == var("ink")
    from backend import config
    assert cfg["server"]["maxUploadSize"] == config.MAX_FILE_MB    # upload box must not promise more than the API accepts


# ------------------------------------------------------------------ password gate and report download
def test_password_gate(monkeypatch, result):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = run_app({"result": result})                               # a result is waiting, but the gate comes first
    assert len(at.text_input) == 1 and 'class="report' not in markdown(at)

    at.text_input[0].set_value("wrong").run()
    assert at.error and 'class="report' not in markdown(at)

    at.text_input[0].set_value("s3cret").run()
    assert not at.exception and 'class="report' in markdown(at)    # unlocked


def test_no_password_means_open(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    at = run_app()
    assert not at.exception and len(at.text_input) == 0 and len(at.text_area) == 1   # no gate, just the form


def test_markdown_report(result):
    md = app_helpers()["to_markdown"](result)
    assert md.startswith("# ATS Report") and "Kubernetes" in md
