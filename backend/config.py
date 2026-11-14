"""All settings live here. Values come from environment variables or a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Groq retired llama-3.3-70b-versatile on 2026-08-16, so we default to gpt-oss-120b.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

MAX_FILE_MB = 5
MAX_LLM_CHARS = 12000      # resume / JD text sent to the LLM is cut to this length
MIN_TEXT_CHARS = 80        # below this we assume the PDF is a scanned image
