"""Grounded suggestions: check what the LLM is shown and what happens when it fails."""
import pytest

from backend import analyzer, llm_parser
from backend.llm_parser import LLMError
from backend.rag import build_index, parsed_to_documents
from backend.schemas import JDMatch, ParsedResume, Suggestion
from backend.suggestions import SuggestionList, generate_suggestions, weak_bullets
from tests.conftest import SAMPLE_TEXT


@pytest.fixture
def spy(monkeypatch):
    seen = {}

    def fake_run(system, human, schema, **variables):
        seen.update(system=system, **variables)
        return SuggestionList(items=[Suggestion(target="Kubernetes", advice="Try minikube", example_bullet="")])

    monkeypatch.setattr(llm_parser, "run_structured", fake_run)
    return seen


def test_jd_mode_sends_retrieved_evidence_not_the_whole_resume(spy, parsed_resume):
    index = build_index(parsed_to_documents(parsed_resume))
    jd_match = JDMatch(match_percentage=50, keyword_match_pct=50, skills_gap=["Kubernetes"])
    out = generate_suggestions(index, parsed_resume, jd_match, "Backend Engineer")
    ctx = spy["context"]
    assert "TARGET: Kubernetes" in ctx and "EVIDENCE:" in ctx and "similarity" in ctx
    assert "[Project: " in ctx or "[Experience: " in ctx          # retrieved lines carry their source
    assert "Never invent" in spy["system"] and spy["role"] == "Backend Engineer"
    assert out[0].target == "Kubernetes"


def test_no_jd_mode_targets_weak_bullets(spy, parsed_resume):
    out = generate_suggestions(None, parsed_resume, None)
    assert "Responsible for writing documentation" in spy["context"]
    assert out


def test_nothing_to_improve_means_no_llm_call(monkeypatch):
    monkeypatch.setattr(llm_parser, "run_structured", lambda *a, **k: pytest.fail("LLM must not be called"))
    p = ParsedResume()
    assert generate_suggestions(None, p, None) == []


def test_weak_bullets_prefers_weak_openers(parsed_resume):
    assert weak_bullets(parsed_resume)[0].startswith("Responsible")


def test_analyzer_turns_llm_failure_into_a_warning(monkeypatch, parsed_resume, parsed_jd):
    monkeypatch.setattr(analyzer, "parse_resume", lambda t: parsed_resume)
    monkeypatch.setattr(analyzer, "parse_job_description", lambda t: parsed_jd)

    def boom(*a, **k):
        raise LLMError("rate limited")

    monkeypatch.setattr(llm_parser, "run_structured", boom)
    r = analyzer.analyze(SAMPLE_TEXT, "jd")
    assert r.suggestions == [] and any("rate limited" in w for w in r.warnings)
    assert r.ats_score > 0                                          # the rest of the analysis survived


def test_analyzer_survives_embedding_failure(monkeypatch, parsed_resume, parsed_jd):
    monkeypatch.setattr(analyzer, "parse_resume", lambda t: parsed_resume)
    monkeypatch.setattr(analyzer, "parse_job_description", lambda t: parsed_jd)
    monkeypatch.setattr(analyzer, "build_index", lambda docs: (_ for _ in ()).throw(OSError("no internet")))
    monkeypatch.setattr(analyzer, "generate_suggestions", lambda *a, **k: [])
    r = analyzer.analyze(SAMPLE_TEXT, "jd")
    assert any("Semantic search is unavailable" in w for w in r.warnings)
    assert r.jd_match.semantic_similarity is None and r.ats_score > 0
