"""Grounded AI suggestions: retrieve real resume lines first, then let the LLM rewrite them.

The LLM only sees the retrieved evidence, and is told not to invent anything.
"""
from typing import Optional

from backend import llm_parser
from backend.rag import retrieve
from backend.schemas import JDMatch, Lenient, ParsedResume, Suggestion


class SuggestionList(Lenient):
    items: list[Suggestion] = []


SYSTEM = (
    "You are a careful resume coach for entry-level candidates. "
    "For each TARGET below, write one suggestion using ONLY the EVIDENCE lines shown; they are real lines "
    "from the candidate's resume, each with a similarity score (higher = more related). "
    "Never invent employers, tools, results or experience. "
    "If the evidence is related, rewrite ONE real line so it honestly highlights the target and put it in "
    "example_bullet. Do not add new numbers: if a number is missing, use a placeholder like [X%]. "
    "If no evidence is related, give one small realistic step (such as a mini project) and leave "
    "example_bullet empty. Keep each advice under 40 words. Return JSON with an 'items' list; "
    "'target' must repeat the TARGET text."
)
HUMAN = "Target role: {role}\n\n{context}"
_WEAK_STARTS = ("responsible", "worked", "helped", "assisted", "involved", "handled", "tasked")


def weak_bullets(parsed: ParsedResume, limit: int = 3) -> list[str]:
    lines = [ln.strip() for d in [e.description for e in parsed.experience] + [p.description for p in parsed.projects]
             for ln in d.splitlines() if len(ln.strip()) > 15]
    weak = [b for b in lines if b.lower().startswith(_WEAK_STARTS)]
    no_numbers = [b for b in lines if b not in weak and not any(ch.isdigit() for ch in b)]
    return (weak + no_numbers)[:limit]


def build_context(index, targets: list[str], k: int = 2) -> str:
    blocks = []
    for target in targets:
        hits = retrieve(index, f"experience with {target}", k=k)
        lines = "\n".join(f"- [{h.source}] (similarity {h.score:.2f}) {h.text[:300]}" for h in hits)
        blocks.append(f"TARGET: {target}\nEVIDENCE:\n{lines or '- (nothing in the resume)'}")
    return "\n\n".join(blocks)


def generate_suggestions(index, parsed: ParsedResume, jd_match: Optional[JDMatch], role: str = "") -> list[Suggestion]:
    gap = (jd_match.skills_gap or jd_match.missing_keywords)[:4] if jd_match else []
    if gap:                                                   # mode 1: what the job wants but the resume lacks
        context = build_context(index, gap)
    else:                                                     # mode 2: no JD -> strengthen the weakest bullets
        bullets = weak_bullets(parsed)
        if not bullets:
            return []
        context = "\n\n".join(
            f"TARGET: Make this bullet stronger and measurable\nEVIDENCE:\n- {b}" for b in bullets)
    result = llm_parser.run_structured(SYSTEM, HUMAN, SuggestionList, role=role or "not specified", context=context)
    return result.items[:4]
