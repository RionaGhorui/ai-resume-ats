"""Compare a parsed resume with a parsed job description (keywords + RAG evidence)."""
from backend import config
from backend.matching import match_terms, normalize
from backend.rag import retrieve
from backend.schemas import JDMatch, ParsedJD, ParsedResume, RequirementMatch


def _scale(sim: float) -> float:
    """Map a raw cosine similarity to 0..1 (unrelated text scores ~0.1, a strong match ~0.5+)."""
    return max(0.0, min(1.0, (sim - config.SIM_FLOOR) / (config.SIM_CEIL - config.SIM_FLOOR)))


def match_jd(resume_text: str, parsed: ParsedResume, jd: ParsedJD, index=None) -> JDMatch:
    jd_terms = jd.required_skills + jd.preferred_skills + jd.keywords
    matched, missing = match_terms(jd_terms, parsed.skills + parsed.keywords, resume_text)
    total = len(matched) + len(missing)
    keyword_pct = round(100 * len(matched) / total, 1) if total else 0.0
    required = {normalize(s) for s in jd.required_skills}
    gap = [t for t in missing if normalize(t) in required]

    # RAG part: for every duty in the JD, find the resume entry that best supports it
    matches: list[RequirementMatch] = []
    semantic = None
    if index is not None:
        queries = jd.responsibilities[:6] or [f"experience with {s}" for s in jd.required_skills[:6]]
        for q in queries:
            hit = retrieve(index, q, k=1)
            if hit:
                matches.append(RequirementMatch(requirement=q, source=hit[0].source, score=round(hit[0].score, 2)))
        if matches:
            semantic = round(sum(_scale(m.score) for m in matches) / len(matches), 3)

    pct = keyword_pct if semantic is None else round(0.6 * keyword_pct + 0.4 * 100 * semantic, 1)
    return JDMatch(
        match_percentage=pct, keyword_match_pct=keyword_pct, semantic_similarity=semantic,
        matched_keywords=matched, missing_keywords=missing, skills_gap=gap, requirement_matches=matches,
    )
