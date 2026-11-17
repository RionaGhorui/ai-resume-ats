"""MCP tests: in-process for logic, plus one real stdio subprocess to prove the protocol stream is clean."""
import asyncio
import os
import sys
from pathlib import Path

import pytest
from mcp import StdioServerParameters
from mcp.client import Client

from backend import analyzer
from backend.mcp_server import mcp
from tests.conftest import SAMPLES

ROOT = Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


def test_tools_are_listed():
    async def go():
        async with Client(mcp) as c:
            return {t.name for t in (await c.list_tools()).tools}

    assert run(go()) == {"score_resume", "score_resume_file", "find_resume_evidence"}


def test_find_resume_evidence_needs_no_llm(resume_text):
    async def go():
        async with Client(mcp) as c:
            return await c.call_tool("find_resume_evidence",
                                     {"resume_text": resume_text, "query": "docker", "k": 2})

    r = run(go())
    hits = r.structured_content["result"]
    assert not r.is_error and len(hits) == 2
    assert {"chunk", "text", "similarity"} <= set(hits[0])


def test_score_resume_file_runs_the_full_pipeline(monkeypatch, parsed_resume, parsed_jd):
    monkeypatch.setattr(analyzer, "parse_resume", lambda t: parsed_resume)
    monkeypatch.setattr(analyzer, "parse_job_description", lambda t: parsed_jd)
    monkeypatch.setattr(analyzer, "generate_suggestions", lambda *a, **k: [])

    async def go():
        async with Client(mcp) as c:
            return await c.call_tool("score_resume_file", {
                "path": str(SAMPLES / "sample_resume.pdf"), "job_description": "python backend"})

    r = run(go())
    body = r.structured_content
    assert not r.is_error and 0 < body["ats_score"] <= 100 and body["jd_match"]["skills_gap"]


@pytest.mark.parametrize("path,message", [
    ("/definitely/not/here.pdf", "File not found"),
    (str(SAMPLES / "sample_jd.txt"), "Only .pdf and .docx"),
])
def test_errors_reach_the_client_with_their_message(path, message):
    async def go():
        async with Client(mcp) as c:
            return await c.call_tool("score_resume_file", {"path": path})

    r = run(go())
    assert r.is_error and message in r.content[0].text


def test_missing_api_key_is_explained(monkeypatch, resume_text):
    monkeypatch.setattr("backend.config.GROQ_API_KEY", "")

    async def go():
        async with Client(mcp) as c:
            return await c.call_tool("score_resume", {"resume_text": resume_text})

    r = run(go())
    assert r.is_error and "GROQ_API_KEY" in r.content[0].text


def test_real_stdio_subprocess(resume_text):
    """Start `python -m backend.mcp_server` exactly like Claude Desktop does."""
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "backend.mcp_server"],
        env={**os.environ, "EMBEDDINGS_BACKEND": "fake", "PYTHONPATH": str(ROOT)},
    )

    async def go():
        async with Client(params) as c:
            tools = (await c.list_tools()).tools
            r = await c.call_tool("find_resume_evidence",
                                  {"resume_text": resume_text, "query": "pytest", "k": 1})
            return [t.name for t in tools], r

    names, r = asyncio.run(asyncio.wait_for(go(), timeout=60))
    assert "score_resume" in names and len(r.structured_content["result"]) == 1
