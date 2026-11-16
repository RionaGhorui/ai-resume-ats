"""FastAPI app. Run with:  uvicorn backend.main:app --port 8000"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from backend import config
from backend.analyzer import analyze
from backend.llm_parser import LLMError
from backend.models import get_embeddings
from backend.parsing import ResumeFileError, extract_text
from backend.schemas import AnalysisResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ats")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_embeddings()                       # load the model now so the first request is not slow
    except Exception:
        log.warning("Embedding model not ready; it will be retried on the first request.", exc_info=True)
    yield


app = FastAPI(title="AI Resume ATS", version="3.0.0", lifespan=lifespan)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "llm_configured": bool(config.GROQ_API_KEY), "model": config.GROQ_MODEL,
            "embeddings": config.EMBEDDINGS_BACKEND}


# A plain `def` (not `async def`): FastAPI runs it in a worker thread, so the slow
# blocking calls inside (LLM, parsing) do not freeze the server for other users.
@app.post("/api/v1/analyze-resume", response_model=AnalysisResponse)
def analyze_resume(resume: UploadFile = File(...), job_description: str = Form("")):
    data = resume.file.read(config.MAX_FILE_MB * 1024 * 1024 + 1)
    try:
        text = extract_text(data, resume.filename or "")
        return analyze(text, job_description)
    except ResumeFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception:
        log.exception("analysis failed")
        raise HTTPException(status_code=500, detail="Unexpected error while analysing the resume.")
