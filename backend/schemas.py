"""Pydantic models: (1) what the LLM extracts, (2) what the API returns."""
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, BeforeValidator, model_validator


# ---------- helpers: LLMs sometimes return null or the wrong type ----------
def _str_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    return [str(x).strip() for x in v if x is not None and str(x).strip()]


def _to_int(v):
    try:
        return max(0, int(float(v)))
    except (TypeError, ValueError):
        return 0


StrList = Annotated[list[str], BeforeValidator(_str_list)]
Months = Annotated[int, BeforeValidator(_to_int)]


class Lenient(BaseModel):
    """Drop null values so defaults apply instead of crashing validation."""

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data):
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


# ---------- 1. LLM extraction schemas ----------
class Experience(Lenient):
    job_title: str = ""
    company: str = ""
    start_date: str = ""
    end_date: str = ""
    duration_months: Months = 0
    description: str = ""          # all bullet points, joined by newlines


class Education(Lenient):
    degree: str = ""
    institution: str = ""
    year: str = ""
    gpa: str = ""


class Project(Lenient):
    title: str = ""
    description: str = ""          # all bullet points, joined by newlines
    technologies: StrList = []


class ParsedResume(Lenient):
    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    professional_summary: str = ""
    skills: StrList = []
    experience: list[Experience] = []
    education: list[Education] = []
    projects: list[Project] = []
    certifications: StrList = []
    action_verbs: StrList = []
    keywords: StrList = []


class ParsedJD(Lenient):
    job_title: str = ""
    required_skills: StrList = []
    preferred_skills: StrList = []
    keywords: StrList = []
    responsibilities: StrList = []


# ---------- 2. API response schemas ----------
class IssueDetail(BaseModel):
    issue_title: str
    severity_level: Literal["High", "Moderate", "Low"]
    ats_impact: Literal["High", "Medium", "Low"]
    explanation: str
    where_it_appears: str
    how_to_fix: str
    action_items: list[str]
    example_improvement: str


class ComponentScore(BaseModel):
    name: str
    score: float
    max_score: float


class SkillEvidence(BaseModel):
    skill: str
    evidence: list[str] = []       # e.g. "Project: Expense Tracker API"


class SkillValidation(BaseModel):
    validated: list[SkillEvidence] = []
    unvalidated: list[str] = []
    validation_pct: float = 0.0


class RequirementMatch(BaseModel):
    requirement: str               # a duty from the job description
    source: str                    # best-matching part of the resume
    score: float                   # cosine similarity (0-1)


class JDMatch(BaseModel):
    match_percentage: float
    keyword_match_pct: float
    semantic_similarity: Optional[float] = None   # filled in from Phase 3 (RAG)
    matched_keywords: list[str] = []
    missing_keywords: list[str] = []
    skills_gap: list[str] = []                    # required JD skills missing from the resume
    requirement_matches: list[RequirementMatch] = []   # RAG evidence per JD duty (Phase 3)


class Suggestion(Lenient):
    target: str = ""
    advice: str = ""
    example_bullet: str = ""


class AnalysisResponse(BaseModel):
    ats_score: float
    interpretation: str
    component_scores: list[ComponentScore]
    strengths: list[str] = []
    issues_summary: list[str] = []
    detailed_feedback: list[IssueDetail] = []
    skills: list[str] = []
    skill_validation: SkillValidation
    jd_match: Optional[JDMatch] = None
    suggestions: list[Suggestion] = []
    warnings: list[str] = []
