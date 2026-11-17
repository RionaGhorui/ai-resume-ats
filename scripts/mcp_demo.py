"""A tiny MCP *client*: starts the server as a subprocess, lists its tools, calls one.

    python scripts/mcp_demo.py                 # first run downloads the embedding model (~90 MB)
    EMBEDDINGS_BACKEND=fake python scripts/mcp_demo.py   # offline: runs, but similarity numbers are random
"""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp import StdioServerParameters  # noqa: E402
from mcp.client import Client  # noqa: E402

from backend.parsing import extract_text  # noqa: E402


async def main() -> None:
    resume = ROOT / "sample_data" / "sample_resume.docx"
    text = extract_text(resume.read_bytes(), resume.name)
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "backend.mcp_server"],
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    async with Client(server) as client:
        print("Tools offered by the server:")
        for tool in (await client.list_tools()).tools:
            print(f"  - {tool.name}: {tool.description.splitlines()[0]}")

        query = "experience with Docker and containers"
        result = await client.call_tool("find_resume_evidence", {"resume_text": text, "query": query, "k": 2})
        print(f"\nBest resume chunks for: {query!r}")
        for hit in result.structured_content["result"]:
            print(f"  [{hit['similarity']:+.2f}] {hit['chunk']}: {hit['text'][:90]!r}")


if __name__ == "__main__":
    asyncio.run(main())
