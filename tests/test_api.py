import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.llm_parser import LLMError
from backend.schemas import Suggestion
from tests.conftest import SAMPLES


@pytest.fixture
def client(monkeypatch, parsed_resume, parsed_jd):
    # the analyzer calls these two; swap them for hand-written results (no network)
    monkeypatch.setattr("backend.analyzer.parse_resume", lambda text: parsed_resume)
    monkeypatch.setattr("backend.analyzer.parse_job_description", lambda text: parsed_jd)
    monkeypatch.setattr("backend.analyzer.generate_suggestions",
                        lambda *a, **k: [Suggestion(target="Kubernetes", advice="Try minikube")])
    return TestClient(main.app)


def _upload(client, name="sample_resume.docx", jd=""):
    return client.post("/api/v1/analyze-resume",
                       files={"resume": (name, (SAMPLES / name).read_bytes())},
                       data={"job_description": jd})


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["embeddings"] == "fake"


def test_analyze_without_jd(client):
    r = _upload(client)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["ats_score"] <= 100 and body["jd_match"] is None
    assert body["strengths"] and body["detailed_feedback"]       # strengths are no longer dropped


def test_analyze_with_jd_and_pdf(client):
    r = _upload(client, "sample_resume.pdf", jd=(SAMPLES / "sample_jd.txt").read_text())
    assert r.status_code == 200
    body = r.json()
    assert "Kubernetes" in body["jd_match"]["skills_gap"]
    assert body["suggestions"][0]["target"] == "Kubernetes"
    assert body["jd_match"]["semantic_similarity"] is not None and body["jd_match"]["requirement_matches"]


def test_bad_file_gives_400(client):
    r = client.post("/api/v1/analyze-resume", files={"resume": ("x.txt", b"hello")})
    assert r.status_code == 400


def test_llm_failure_gives_502(monkeypatch):
    def boom(text):
        raise LLMError("Groq request failed")
    monkeypatch.setattr("backend.analyzer.parse_resume", boom)
    r = _upload(TestClient(main.app))
    assert r.status_code == 502


def test_jd_failure_is_only_a_warning(monkeypatch, parsed_resume):
    def boom(text):
        raise LLMError("nope")
    monkeypatch.setattr("backend.analyzer.parse_resume", lambda t: parsed_resume)
    monkeypatch.setattr("backend.analyzer.parse_job_description", boom)
    monkeypatch.setattr("backend.analyzer.generate_suggestions", lambda *a, **k: [])
    r = _upload(TestClient(main.app), jd="some jd")
    assert r.status_code == 200 and r.json()["warnings"]


def test_startup_warms_up_embeddings_and_survives_failure(monkeypatch):
    called = []
    monkeypatch.setattr("backend.main.get_embeddings", lambda: called.append(1))
    with TestClient(main.app) as c:                 # `with` runs the lifespan (startup) code
        assert c.get("/api/v1/health").status_code == 200
    assert called == [1]

    def broken():
        raise OSError("no internet")

    monkeypatch.setattr("backend.main.get_embeddings", broken)
    with TestClient(main.app) as c:                 # a failed model download must not stop the server
        assert c.get("/api/v1/health").status_code == 200
