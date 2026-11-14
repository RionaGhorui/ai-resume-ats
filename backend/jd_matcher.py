"""Compare a parsed resume with a parsed job description."""
from backend.matching import match_terms, normalize
from backend.schemas import JDMatch, ParsedJD, ParsedResume


def match_jd(resume_text: str, parsed: ParsedResume, jd: ParsedJD) -> JDMatch:
    jd_terms = jd.required_skills + jd.preferred_skills + jd.keywords
    matched, missing = match_terms(jd_terms, parsed.skills + parsed.keywords, resume_text)
    total = len(matched) + len(missing)
    keyword_pct = round(100 * len(matched) / total, 1) if total else 0.0

    required = {normalize(s) for s in jd.required_skills}
    gap = [t for t in missing if normalize(t) in required]
    return JDMatch(
        match_percentage=keyword_pct,
        keyword_match_pct=keyword_pct,
        matched_keywords=matched,
        missing_keywords=missing,
        skills_gap=gap,
    )
