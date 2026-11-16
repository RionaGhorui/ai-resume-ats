"""RAG tests. ConceptEmbeddings makes related words land close together, so we can check the
retrieval logic (ranking, thresholds, fallbacks) exactly, without downloading a real model."""
import re
import zlib

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from backend import config, models
from backend.jd_matcher import match_jd
from backend.rag import build_index, parsed_to_documents, retrieve, text_to_documents
from backend.schemas import ParsedResume, Project
from backend.skills import validate_skills

CONCEPTS = [
    {"docker", "containerised", "containerized", "container", "containers", "dockerfile"},
    {"api", "endpoint", "endpoints", "rest", "fastapi", "flask"},
    {"test", "tests", "pytest", "coverage"},
    {"model", "classifier", "accuracy", "trained", "scikit-learn"},
]


class ConceptEmbeddings(Embeddings):
    """Related words share a dimension; every other word hashes into one of 64 small buckets,
    so unrelated texts end up far apart (like a real embedding model, but fully deterministic)."""

    def _vec(self, text):
        words = re.findall(r"[a-z\-]+", text.lower())
        concept = [float(sum(w in group for w in words)) for group in CONCEPTS]
        buckets = [0.0] * 64
        for w in words:
            if not any(w in g for g in CONCEPTS):
                buckets[zlib.crc32(w.encode()) % 64] += 0.3
        return concept + buckets

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def concept(monkeypatch):
    models.get_embeddings.cache_clear()
    monkeypatch.setattr("backend.rag.get_embeddings", lambda: ConceptEmbeddings())


def test_retrieval_ranks_the_related_project_first(concept, parsed_resume):
    index = build_index(parsed_to_documents(parsed_resume))
    hits = retrieve(index, "containers", k=2)
    assert hits[0].source == "Project: Expense Tracker API"
    assert hits[0].score > hits[1].score
    top_ml = retrieve(index, "trained a classifier with accuracy", k=1)[0]
    assert top_ml.source == "Project: Movie Review Sentiment Classifier"


def test_semantic_fallback_validates_by_meaning(concept, parsed_resume, monkeypatch):
    # "Containers" never appears as a word (the resume says "Containerised"/"Docker")
    parsed_resume.skills = ["Containers", "Kubernetes"]
    monkeypatch.setattr(config, "SEMANTIC_THRESHOLD", 0.3)
    index = build_index(parsed_to_documents(parsed_resume))

    without_rag = validate_skills(parsed_resume)
    assert "Containers" in without_rag.unvalidated             # exact matching alone misses it

    with_rag = validate_skills(parsed_resume, index)
    ev = {s.skill: s.evidence for s in with_rag.validated}
    assert "Containers" in ev and "meaning match" in ev["Containers"][0]
    assert "Project: Expense Tracker API" in ev["Containers"][0]
    assert "Kubernetes" in with_rag.unvalidated                # unrelated skills are not validated


def test_threshold_is_respected(concept, parsed_resume, monkeypatch):
    parsed_resume.skills = ["Containers"]
    monkeypatch.setattr(config, "SEMANTIC_THRESHOLD", 0.99)
    index = build_index(parsed_to_documents(parsed_resume))
    assert validate_skills(parsed_resume, index).unvalidated == ["Containers"]


def test_jd_match_returns_evidence_per_duty(concept, resume_text, parsed_resume, parsed_jd):
    index = build_index(parsed_to_documents(parsed_resume))
    plain = match_jd(resume_text, parsed_resume, parsed_jd)
    rag = match_jd(resume_text, parsed_resume, parsed_jd, index)
    assert plain.semantic_similarity is None and plain.requirement_matches == []
    assert rag.semantic_similarity is not None and len(rag.requirement_matches) == 4
    deploy = next(m for m in rag.requirement_matches if m.requirement.startswith("Deploy"))
    assert deploy.source == "Project: Expense Tracker API"    # the container project backs this duty
    assert rag.match_percentage != plain.match_percentage


def test_empty_resume_has_no_index():
    assert build_index(parsed_to_documents(ParsedResume())) is None
    assert retrieve(None, "anything") == []


def test_long_entries_are_split_and_keep_their_source():
    p = ParsedResume(projects=[Project(title="Big", description=("Built a thing. " * 200))])
    docs = parsed_to_documents(p)
    assert len(docs) > 1 and all(d.metadata["source"] == "Project: Big" for d in docs)
    assert all(len(d.page_content) <= 500 for d in docs)


def test_raw_text_chunks_are_numbered(resume_text):
    docs = text_to_documents(resume_text * 2)
    assert docs[0].metadata["source"] == "Resume chunk 1" and len(docs) >= 2


def test_fake_backend_is_offline_and_repeatable():
    e = models.get_embeddings()
    assert e.embed_query("hello") == e.embed_query("hello")


def test_fastembed_adapter_shapes(monkeypatch):
    """Check our adapter against fastembed's documented interface, using a stand-in model."""
    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir): self.args = (model_name, cache_dir)
        def embed(self, texts): return (np.arange(3, dtype=np.float32) + len(t) for t in texts)

    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)
    emb = models.FastEmbedEmbeddings("some/model", "/tmp/cache")
    assert emb._model.args == ("some/model", "/tmp/cache")
    docs = emb.embed_documents(["a", "bb"])
    assert docs == [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]] and all(isinstance(x, float) for x in docs[0])
    assert emb.embed_query("a") == [1.0, 2.0, 3.0]
