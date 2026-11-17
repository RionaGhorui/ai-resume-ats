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

## Phase 2: LangChain

`backend/llm_parser.py` now builds a LangChain chain: `ChatPromptTemplate | ChatGroq.with_structured_output(PydanticModel)`.

| | Phase 1 (plain Groq SDK) | Phase 2 (LangChain) |
|---|---|---|
| Output format | JSON schema pasted into the prompt by hand | `with_structured_output(ParsedResume)` |
| Parsing | `json.loads` + manual validation | returns a validated Pydantic object |
| Retry | hand-written loop | `.with_retry(...)` on bad output only |

`run_structured(system, human, schema, **vars)` is the one reusable function. Phase 3 uses it again for AI suggestions.

If you switch `GROQ_MODEL` to a model without JSON-schema support, set `GROQ_STRUCTURED_METHOD=function_calling` in `.env`.

After unzipping Phase 2 in a new folder, run `pip install -r requirements.txt` again (the Groq SDK is now installed through `langchain-groq`).

## Phase 3: RAG (retrieval-augmented generation)

RAG here means: *find the relevant lines of the resume first, then reason about only those lines.*

```
parsed resume ──► chunks (projects, experience, certificates; 500 chars)
                     │  embed with all-MiniLM-L6-v2 (fastembed / ONNX)
                     ▼
              InMemoryVectorStore  (built once per analysis)
                     │
   ┌─────────────────┼──────────────────────────┐
   ▼                 ▼                          ▼
skill validation   JD match                  AI suggestions
"experience using  best resume evidence      retrieved lines ──► LLM
 Docker?"          for each JD duty          (told: never invent)
```

| Where | Before (Phases 1-2) | With RAG |
|---|---|---|
| `skills.py` | exact word match only | exact match, then retrieval by meaning (`Containers` is backed by "Containerised … with Docker") |
| `jd_matcher.py` | keyword overlap only | 60% keywords + 40% meaning; table of best evidence per JD duty |
| `suggestions.py` (new) | none | LLM rewrites real lines to fit the job; missing numbers become `[X%]` placeholders |

New files: `models.py` (embedding loader), `rag.py` (chunk / index / retrieve), `suggestions.py`.
Embeddings use `fastembed` (ONNX) instead of `sentence-transformers`: the **same MiniLM model** without installing PyTorch (about 2 GB smaller).

**Good to know**
- First real run downloads the ~90 MB model into `.cache/fastembed/`. Needs internet once.
- The similarity cut-offs (`SEMANTIC_THRESHOLD`, `SIM_FLOOR`, `SIM_CEIL` in `backend/config.py`) are hand-picked starting values, not tuned on real resumes. Run the sample resume, read the **Similarity** column in the *Job match* tab, then adjust. Raise `SEMANTIC_THRESHOLD` if too many skills are marked as demonstrated.
- If the model cannot load, the analysis still works with keyword matching and shows a warning.
- Tests use a fake embedder (`EMBEDDINGS_BACKEND=fake`), so `pytest` never downloads anything.

After unzipping Phase 3 in a new folder, run `pip install -r requirements.txt` again.

## Phase 4: MCP server

MCP (Model Context Protocol) is a standard way for AI apps (Claude Desktop, IDEs, agents) to call your code as *tools*.
`backend/mcp_server.py` exposes the same pipeline the REST API uses. It adds no new scoring logic.

| Tool | Does | Needs `GROQ_API_KEY`? |
|---|---|---|
| `score_resume(resume_text, job_description="")` | full analysis, same JSON as the API | yes |
| `score_resume_file(path, job_description="")` | same, but reads a local `.pdf` / `.docx` | yes |
| `find_resume_evidence(resume_text, query, k=3)` | semantic search inside a resume (the RAG part on its own) | no |

**See it work (no extra software):**

```bash
python scripts/mcp_demo.py                      # starts the server, lists tools, calls one
python -m backend.mcp_server                    # run the server by hand (it waits silently for a client)
```

**Use it from Claude Desktop** (optional). Edit the config file
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`, Windows: `%APPDATA%\Claude\claude_desktop_config.json`)
and use absolute paths. The server reads `GROQ_API_KEY` from your project's `.env`, so the key does not go in this file:

```json
{
  "mcpServers": {
    "ai-resume-ats": {
      "command": "/ABSOLUTE/PATH/ai-resume-ats/.venv/bin/python",
      "args": ["-m", "backend.mcp_server"],
      "env": { "PYTHONPATH": "/ABSOLUTE/PATH/ai-resume-ats" }
    }
  }
}
```

On Windows use `C:\\...\\ai-resume-ats\\.venv\\Scripts\\python.exe`. Restart Claude Desktop, then ask:
*"Score the resume at /path/to/resume.pdf for this job description: ..."*

**Things to know**
- This project uses **mcp 2.x**, where `FastMCP` was renamed to `MCPServer` (`from mcp.server.mcpserver import MCPServer`). Older tutorials show the v1 import and will not work with this version.
- With the stdio transport, stdout carries the protocol, so the server must never `print()`. It logs to stderr.
- Expected failures are raised as `ToolError`, so the client sees the real message ("File not found", "GROQ_API_KEY is not set").
- `score_resume_file` reads any `.pdf`/`.docx` path it is given. That is fine for a local stdio server, but do not expose it over a network as-is.
