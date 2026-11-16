"""The whole pipeline in one function. The API and the MCP server both call this."""
import logging

from backend.feedback import find_issues
from backend.jd_matcher import match_jd
from backend.llm_parser import LLMError, parse_job_description, parse_resume
from backend.rag import build_index, parsed_to_documents
from backend.schemas import AnalysisResponse
from backend.scoring import find_strengths, interpret, score_resume
from backend.skills import validate_skills
from backend.suggestions import generate_suggestions

log = logging.getLogger("ats")


def analyze(resume_text: str, job_description: str = "") -> AnalysisResponse:
    warnings: list[str] = []

    parsed = parse_resume(resume_text)

    jd = None
    if job_description.strip():
        try:
            jd = parse_job_description(job_description)
        except LLMError as exc:               # a failed JD parse should not kill the whole analysis
            warnings.append(f"Job description could not be analysed: {exc}")

    # RAG: index the resume's projects/experience once, reuse for skills, JD match and suggestions
    try:
        index = build_index(parsed_to_documents(parsed))
    except Exception:                         # e.g. the embedding model could not be downloaded
        log.exception("embedding / indexing failed")
        index = None
        warnings.append("Semantic search is unavailable, so only keyword matching was used.")

    validation = validate_skills(parsed, index)
    jd_match = match_jd(resume_text, parsed, jd, index) if jd else None
    components, metrics = score_resume(resume_text, parsed, validation, jd_match)
    issues = find_issues(resume_text, parsed, validation, components, metrics, jd_match)
    total = round(sum(c.score for c in components), 1)

    try:
        suggestions = generate_suggestions(index, parsed, jd_match, jd.job_title if jd else "")
    except LLMError as exc:
        suggestions = []
        warnings.append(f"AI suggestions unavailable: {exc}")

    return AnalysisResponse(
        ats_score=total,
        interpretation=interpret(total),
        component_scores=components,
        strengths=find_strengths(metrics, validation, jd_match),
        issues_summary=[i.issue_title for i in issues],
        detailed_feedback=issues,
        skills=parsed.skills,
        skill_validation=validation,
        jd_match=jd_match,
        suggestions=suggestions,
        warnings=warnings,
    )
