"""The whole pipeline in one function. The API and (later) the MCP server both call this."""
from backend.feedback import find_issues
from backend.jd_matcher import match_jd
from backend.llm_parser import LLMError, parse_job_description, parse_resume
from backend.schemas import AnalysisResponse
from backend.scoring import find_strengths, interpret, score_resume
from backend.skills import validate_skills


def analyze(resume_text: str, job_description: str = "") -> AnalysisResponse:
    warnings: list[str] = []

    parsed = parse_resume(resume_text)

    jd = None
    if job_description.strip():
        try:
            jd = parse_job_description(job_description)
        except LLMError as exc:               # a failed JD parse should not kill the whole analysis
            warnings.append(f"Job description could not be analysed: {exc}")

    validation = validate_skills(parsed)
    jd_match = match_jd(resume_text, parsed, jd) if jd else None
    components, metrics = score_resume(resume_text, parsed, validation, jd_match)
    issues = find_issues(resume_text, parsed, validation, components, metrics, jd_match)
    total = round(sum(c.score for c in components), 1)

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
        warnings=warnings,
    )
