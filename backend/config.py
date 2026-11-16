"""All settings live here. Values come from environment variables or a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Groq retired llama-3.3-70b-versatile on 2026-08-16, so we default to gpt-oss-120b.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# json_schema works on gpt-oss models. If you switch to a model without it, try function_calling.
GROQ_STRUCTURED_METHOD = os.getenv("GROQ_STRUCTURED_METHOD", "json_schema")

MAX_FILE_MB = 5
MAX_LLM_CHARS = 12000      # resume / JD text sent to the LLM is cut to this length
MIN_TEXT_CHARS = 80        # below this we assume the PDF is a scanned image

# ---- RAG (Phase 3) ----
EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "fastembed")        # "fastembed" | "fake" (tests)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", str(ROOT / ".cache" / "fastembed"))
RETRIEVE_K = 2
# Cosine similarity cut-offs tuned by hand for all-MiniLM-L6-v2. Look at real outputs and adjust.
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.55"))
SIM_FLOOR, SIM_CEIL = 0.15, 0.55     # similarity at/below FLOOR counts as 0%, at/above CEIL as 100%
