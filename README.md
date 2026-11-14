# AI Resume ATS

Upload a resume (PDF/DOCX), optionally paste a job description, and get an ATS score,
a skills-vs-evidence check, a job-match report and specific fixes. Built for freshers.

**Stack:** FastAPI · Streamlit · Groq LLM · LangChain · RAG · MCP · Docker

| Phase | Adds | Needs a Groq key? |
|---|---|---|
| 1 | Fixed baseline: parsing, scoring, feedback, API + UI | yes (for resume parsing) |
| 2 | LangChain structured output replaces hand-written JSON parsing | yes |
| 3 | RAG: resume evidence retrieval + grounded AI suggestions | yes |
| 4 | MCP server that exposes the scorer as tools | partly |
| 5 | Docker, CI and cloud deployment | yes |

Get a free key at <https://console.groq.com>. You need **Python 3.10+** (tested on 3.12).

## Setup (every phase)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
python -m compileall -q backend frontend      # "compile" check: prints nothing when OK
python -m pytest -q                           # tests (no API key or internet needed)
cp .env.example .env                          # Windows: copy .env.example .env
# open .env and paste your GROQ_API_KEY
```

## Run

```bash
# terminal 1: API
uvicorn backend.main:app --port 8000
# terminal 2: UI  ->  http://localhost:8501
streamlit run frontend/app.py
```

Try it with `sample_data/sample_resume.pdf` and paste `sample_data/sample_jd.txt` as the job description.

## Phase 1: what changed from the original project

Bugs fixed
- `feedback_engine.py` had a stray first line (a `SyntaxError`), so no analysis could finish. Rewritten as `feedback.py`.
- The default Groq model `llama-3.3-70b-versatile` was shut down on 2026-08-16. Default is now `openai/gpt-oss-120b` (set `GROQ_MODEL` to change).
- The resume parser called the LLM twice on every success (inverted `if result is None`). Now it retries only on failure.
- Grammar and location checks were stubs that always passed (and gave 10 free content points). Replaced by a real writing-quality check and a regex privacy check.
- `strengths` was computed but never returned by the API. Now returned and shown.
- Skill matching used raw substring search (`"go"` matched `"good"`). Now whole-word with aliases (`JS` = `JavaScript`).
- Blocking work ran inside `async def` endpoints. Now plain `def`, which FastAPI runs in worker threads.

Removed to keep it small: spaCy, WeasyPrint PDF export (Markdown/JSON download instead), PyPDF2, python-magic, Supabase auth/history, 12-file UI.

## Layout

```
backend/
  main.py          FastAPI app (health + /api/v1/analyze-resume)
  analyzer.py      the whole pipeline in one function
  parsing.py       PDF/DOCX -> text
  llm_parser.py    Groq: text -> structured resume / JD
  skills.py        are listed skills demonstrated in projects/experience?
  jd_matcher.py    resume vs job description
  scoring.py       5-part rubric = 100 points
  feedback.py      rule-based issues with fixes
  matching.py      aliases + whole-word + fuzzy matching
  schemas.py       Pydantic models
frontend/app.py    Streamlit UI (single file)
tests/             30 tests, LLM is faked
```
