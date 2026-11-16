"""Headless Streamlit test: render the page with a ready-made result (no backend, no LLM)."""
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

from backend import analyzer
from tests.conftest import SAMPLE_TEXT

APP = pathlib.Path(__file__).resolve().parents[1] / "frontend" / "app.py"


@pytest.fixture
def result(monkeypatch, parsed_resume, parsed_jd):
    monkeypatch.setattr(analyzer, "parse_resume", lambda t: parsed_resume)
    monkeypatch.setattr(analyzer, "parse_job_description", lambda t: parsed_jd)
    return analyzer.analyze(SAMPLE_TEXT, "jd").model_dump()


def test_page_loads_empty():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert "AI Resume ATS" in at.title[0].value


def test_page_renders_a_result(result):
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["result"] = result
    at.run()
    assert not at.exception
    assert any("ATS score" in m.label for m in at.metric)
    assert len(at.expander) == len(result["detailed_feedback"])


def test_markdown_report(result):
    # load only the helper functions (everything before show()) without running the page
    src = APP.read_text().split("def show(")[0]
    ns: dict = {}
    exec(compile(src, "app_helpers", "exec"), ns)
    md = ns["to_markdown"](result)
    assert md.startswith("# ATS Report") and "Kubernetes" in md
