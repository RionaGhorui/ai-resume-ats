"""Phase 4: expose the resume scorer as MCP tools (Model Context Protocol).

Run it:   python -m backend.mcp_server        (talks over stdin/stdout)

IMPORTANT: with the stdio transport, stdout belongs to the protocol. Never print() in this
module or in anything it imports. Logging goes to stderr.
"""
import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from backend.analyzer import analyze
from backend.llm_parser import LLMError
from backend.parsing import ResumeFileError, extract_text
from backend.rag import build_index, retrieve, text_to_documents

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

mcp = MCPServer(
    "ai-resume-ats",
    instructions=(
        "Tools to analyse a resume for ATS compatibility. Use score_resume_file for a PDF/DOCX on disk, "
        "score_resume for pasted text, and find_resume_evidence to look up what a resume says about a "
        "topic (no API key needed)."
    ),
)


@mcp.tool()
def score_resume(resume_text: str, job_description: str = "") -> dict[str, Any]:
    """Score a resume (0-100) with feedback, skill evidence and an optional job-description match.

    Needs GROQ_API_KEY. Returns the same JSON as the REST API.
    """
    try:
        return analyze(resume_text, job_description).model_dump()
    except LLMError as exc:
        raise ToolError(str(exc)) from exc        # ToolError = expected failure: the client sees this message


@mcp.tool()
def score_resume_file(path: str, job_description: str = "") -> dict[str, Any]:
    """Same as score_resume, but reads a local .pdf or .docx file from `path`."""
    file = Path(path).expanduser()
    if file.suffix.lower() not in (".pdf", ".docx"):
        raise ToolError("Only .pdf and .docx files are supported.")
    if not file.is_file():
        raise ToolError(f"File not found: {file}")
    try:
        text = extract_text(file.read_bytes(), file.name)
    except ResumeFileError as exc:
        raise ToolError(str(exc)) from exc
    return score_resume(text, job_description)


@mcp.tool()
def find_resume_evidence(resume_text: str, query: str, k: int = 3) -> list[dict[str, Any]]:
    """Semantic search inside a resume: returns the k chunks that best match `query`.

    Example query: "experience with Docker". Uses embeddings only, so no API key is needed.
    """
    index = build_index(text_to_documents(resume_text))
    hits = retrieve(index, query, k=max(1, min(k, 10)))
    return [{"chunk": h.source, "text": h.text, "similarity": round(h.score, 3)} for h in hits]


if __name__ == "__main__":
    mcp.run(transport="stdio")
