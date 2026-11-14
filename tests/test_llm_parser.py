"""LangChain parser tests. No network: the model is replaced by fakes at two different depths."""
import pytest
from groq.resources.chat.completions import Completions
from groq.types.chat import ChatCompletion
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from backend import config, llm_parser
from backend.llm_parser import LLMError
from backend.schemas import ParsedJD, ParsedResume


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")


# ---------- level 1: replace the structured model with a plain function ----------
def test_text_and_date_reach_the_prompt(monkeypatch):
    seen = {}

    def fake_model(prompt_value):
        seen["prompt"] = prompt_value.to_string()
        return ParsedResume(name="A", skills=["Python"])

    monkeypatch.setattr(llm_parser, "_structured_llm", lambda schema: RunnableLambda(fake_model))
    r = llm_parser.parse_resume("MY RESUME {with braces}")
    assert r.name == "A" and r.skills == ["Python"]
    assert "MY RESUME {with braces}" in seen["prompt"]      # braces in resumes must not break the template
    assert "today is 20" in seen["prompt"]


def test_bad_output_is_retried_once(monkeypatch):
    calls = []

    def flaky(prompt_value):
        calls.append(1)
        if len(calls) == 1:
            raise OutputParserException("bad json")
        return ParsedJD(job_title="X")

    monkeypatch.setattr(llm_parser, "_structured_llm", lambda schema: RunnableLambda(flaky))
    assert llm_parser.parse_job_description("jd").job_title == "X" and len(calls) == 2


def test_valid_output_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_parser, "_structured_llm",
                        lambda s: RunnableLambda(lambda pv: calls.append(1) or ParsedResume()))
    llm_parser.parse_resume("x")
    assert len(calls) == 1                                   # the original code always made 2 calls


def test_api_errors_are_wrapped(monkeypatch):
    def boom(prompt_value):
        raise RuntimeError("model_decommissioned")

    monkeypatch.setattr(llm_parser, "_structured_llm", lambda schema: RunnableLambda(boom))
    with pytest.raises(LLMError, match="model_decommissioned"):
        llm_parser.parse_resume("x")


def test_missing_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        llm_parser.parse_resume("x")


# ---------- level 2: run the REAL langchain-groq code, fake only the HTTP call ----------
def _completion(content: str) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
    })


def test_real_langchain_groq_json_schema_path(monkeypatch):
    captured = {}

    def fake_create(self, **kwargs):
        captured.update(kwargs)
        # note the nulls and the string-typed number: the Lenient schemas must absorb them
        return _completion('{"name": "Ada", "skills": ["Python", null], "projects": null, '
                           '"experience": [{"job_title": "Intern", "duration_months": "3"}]}')

    monkeypatch.setattr(Completions, "create", fake_create)
    r = llm_parser.parse_resume("some resume text")
    assert r.name == "Ada" and r.skills == ["Python"] and r.projects == []
    assert r.experience[0].duration_months == 3
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["model"] == config.GROQ_MODEL
    assert captured.get("reasoning_effort") == "low"          # gpt-oss default
    assert any("some resume text" in str(m) for m in captured["messages"])


def test_function_calling_method_also_builds(monkeypatch):
    monkeypatch.setattr(config, "GROQ_STRUCTURED_METHOD", "function_calling")
    assert llm_parser._structured_llm(ParsedResume) is not None
