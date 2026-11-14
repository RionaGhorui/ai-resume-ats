"""The Groq client is faked, so this tests our retry/validation logic only."""
import pytest

from backend import config, llm_parser
from backend.llm_parser import LLMError


class _Msg:
    def __init__(self, content): self.content = content


class _Resp:
    def __init__(self, content): self.choices = [type("C", (), {"message": _Msg(content)})()]


def fake_client(answers):
    calls = []

    class Client:
        def __init__(self, **kw): pass

        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    calls.append(kw)
                    a = answers.pop(0)
                    if isinstance(a, Exception):
                        raise a
                    return _Resp(a)
    return Client, calls


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")


def test_valid_answer_makes_exactly_one_call(monkeypatch):
    Client, calls = fake_client(['{"name": "A", "skills": ["Python"], "projects": null}'])
    monkeypatch.setattr(llm_parser, "Groq", Client)
    r = llm_parser.parse_resume("resume text")
    assert r.name == "A" and r.skills == ["Python"] and r.projects == []   # null tolerated
    assert len(calls) == 1                                                  # the old code made 2


def test_bad_json_is_retried_once(monkeypatch):
    Client, calls = fake_client(["not json", '{"job_title": "X"}'])
    monkeypatch.setattr(llm_parser, "Groq", Client)
    assert llm_parser.parse_job_description("jd").job_title == "X" and len(calls) == 2


def test_two_bad_answers_raise(monkeypatch):
    Client, _ = fake_client(["nope", "still nope"])
    monkeypatch.setattr(llm_parser, "Groq", Client)
    with pytest.raises(LLMError):
        llm_parser.parse_resume("x")


def test_api_error_is_wrapped(monkeypatch):
    Client, _ = fake_client([RuntimeError("model_decommissioned")])
    monkeypatch.setattr(llm_parser, "Groq", Client)
    with pytest.raises(LLMError, match="model_decommissioned"):
        llm_parser.parse_resume("x")


def test_missing_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        llm_parser.parse_resume("x")
