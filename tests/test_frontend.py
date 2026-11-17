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


def test_password_gate(monkeypatch, result):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["result"] = result            # even with a result waiting, the gate must come first
    at.run()
    assert len(at.text_input) == 1 and len(at.metric) == 0       # locked: no results visible

    at.text_input[0].set_value("wrong").run()
    assert at.error and len(at.metric) == 0

    at.text_input[0].set_value("s3cret").run()
    assert not at.exception and len(at.metric) == 3              # unlocked: the dashboard renders


def test_no_password_means_open(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception and len(at.text_input) == 0
