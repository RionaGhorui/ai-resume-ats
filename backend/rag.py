"""A tiny RAG pipeline over ONE resume: chunk -> embed -> store -> retrieve.

The "documents" are the resume's own projects / experience / certificates, so retrieval
returns the *evidence* for a question such as "experience using Docker".
"""
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend import config
from backend.models import get_embeddings
from backend.schemas import ParsedResume
from backend.skills import evidence_sources

_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)


@dataclass
class Evidence:
    source: str      # e.g. "Project: Expense Tracker API"
    text: str
    score: float     # cosine similarity, roughly 0..1


def parsed_to_documents(parsed: ParsedResume) -> list[Document]:
    docs = [Document(page_content=text, metadata={"source": label})
            for label, text in evidence_sources(parsed)]
    return _splitter.split_documents(docs)          # long entries are split, metadata is kept


def text_to_documents(text: str) -> list[Document]:
    """Chunk raw resume text (used by the MCP tool, which does not call the LLM)."""
    chunks = _splitter.split_text(text)
    return [Document(page_content=c, metadata={"source": f"Resume chunk {i + 1}"})
            for i, c in enumerate(chunks)]


def build_index(docs: list[Document]):
    """Embed the documents once and keep them in memory. Returns None if there is nothing to index."""
    return InMemoryVectorStore.from_documents(docs, get_embeddings()) if docs else None


def retrieve(index, query: str, k: int | None = None) -> list[Evidence]:
    if index is None:
        return []
    hits = index.similarity_search_with_score(query, k=k or config.RETRIEVE_K)
    return [Evidence(d.metadata.get("source", ""), d.page_content, float(s)) for d, s in hits]
