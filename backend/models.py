"""Embedding model loader (loaded once, on first use)."""
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from backend import config


class FastEmbedEmbeddings(Embeddings):
    """Tiny LangChain adapter around fastembed (ONNX runtime: no PyTorch, small image)."""

    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding   # imported here so tests / MCP start fast

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    if config.EMBEDDINGS_BACKEND == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=256)        # random but repeatable: no model download
    return FastEmbedEmbeddings(config.EMBEDDING_MODEL, config.EMBEDDING_CACHE_DIR)
