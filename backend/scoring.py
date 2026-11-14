"""Rule-based ATS score. Five components that add up to 100:
formatting 20 + keywords 25 + content 25 + skill validation 15 + ATS compatibility 15."""
import re
from typing import Optional

from backend.schemas import ComponentScore, JDMatch, ParsedResume, SkillValidation

SECTION_HEADERS = {
    "experience": {"experience", "work experience", "professional experience", "internships",
                   "internship", "employment", "work history"},
    "projects": {"projects", "academic projects", "personal projects", "key projects"},
    "education": {"education", "academic background", "academics", "qualifications"},
    "skills": {"skills", "technical skills", "core competencies", "key skills"},
    "summary": {"summary", "professional summary", "objective", "career objective", "profile", "about me"},
}
STRONG_VERBS = {"built", "led", "ran", "wrote", "drove", "made", "won", "grew", "spearheaded", "architected"}
NOT_STRONG = {"worked", "helped", "assisted", "used", "involved", "handled", "tasked", "responsible"}
WEAK_PHRASES = ["responsible for", "worked on", "helped with", "duties included",
                "involved in", "assisted in", "tasked with"]

BULLET_RE = re.compile(r"^\s*[•\-\*◦▪●■➢➤–·]\s*(.+)$")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z0-9.]+\s){1,4}(?:street|road|rd|avenue|ave|lane|nagar|colony|sector|apartment|flat)\b",
    re.I,
)
PIN_RE = re.compile(r"(?:\bpin(?:\s*code)?\b[:\s-]*|[,\-–]\s*)\d{6}\b|\b[A-Z]{2}\s\d{5}(?:-\d{4})?\b", re.I)
SPECIAL_RE = re.compile("[\u2500-\u257f\u2600-\u27bf\U0001F300-\U0001FAFF]")


def has_header(text: str, name: str) -> bool:
    names = SECTION_HEADERS[name]
    return any(line.strip().strip(":").lower() in names for line in text.splitlines())


def _is_strong(first_word: str) -> bool:
    if first_word in NOT_STRONG:
        return False
    return first_word in STRONG_VERBS or (first_word.endswith("ed") and len(first_word) >= 5)


def collect_metrics(text: str, parsed: ParsedResume) -> dict:
    """Facts about the resume that both the scorer and the feedback engine use."""
    bullets = [m.group(1).strip() for line in text.splitlines() if (m := BULLET_RE.match(line))]
    if len(bullets) < 3:  # PDF lost the bullet markers: fall back to the parsed descriptions
        desc = [e.description for e in parsed.experience] + [p.description for p in parsed.projects]
        bullets = [ln.strip() for d in desc for ln in d.splitlines() if len(ln.strip()) > 20]
    low = text.lower()
    n = len(bullets)
    firsts = [(re.split(r"\W+", b.lower())[0] if b else "") for b in bullets]
    return {
        "bullets": n,
        "strong_ratio": sum(_is_strong(f) for f in firsts) / n if n else 0.0,
        "quantified_ratio": sum(bool(re.search(r"\d", b)) for b in bullets) / n if n else 0.0,
        "weak_phrases": [p for p in WEAK_PHRASES if p in low],
        "first_person": sum(bool(re.search(r"\b(i|my|me)\b", b.lower())) for b in bullets),
        "long_bullets": sum(len(b) > 250 for b in bullets),
        "words": len(text.split()),
        "has_email": bool(EMAIL_RE.search(text)),
        "has_phone": any(len(re.sub(r"\D", "", m)) >= 10 for m in PHONE_RE.findall(text)),
        "has_linkedin": bool(re.search(r"linkedin\.com/in/[\w-]+", low)),
        "has_github": bool(re.search(r"github\.com/[\w-]+", low)),
        "has_address": bool(ADDRESS_RE.search(text) or PIN_RE.search(text)),
        "special_chars": len(SPECIAL_RE.findall(text)),
    }


def _formatting(text: str, parsed: ParsedResume, m: dict) -> float:
    pts = 0.0
    pts += 4 if has_header(text, "education") or parsed.education else 0
    pts += 4 if has_header(text, "skills") or parsed.skills else 0
    pts += 4 if has_header(text, "projects") or parsed.projects else 0
    pts += 3 if has_header(text, "experience") or parsed.experience else 0
    pts += 1 if has_header(text, "summary") or parsed.professional_summary else 0
    pts += 2 if m["bullets"] >= 4 else (1 if m["bullets"] >= 1 else 0)
    pts += 2 if 200 <= m["words"] <= 1000 else (1 if 100 <= m["words"] <= 1500 else 0)
    return pts


def _keywords(parsed: ParsedResume, jd_match: Optional[JDMatch]) -> float:
    techs = {t.lower() for p in parsed.projects for t in p.technologies}
    generic = (min(len(parsed.skills), 12) / 12 * 12
               + min(len(parsed.keywords), 20) / 20 * 8
               + min(len(techs), 8) / 8 * 5)                 # max 25
    if jd_match is None:
        return generic
    return 25 * (0.7 * jd_match.match_percentage / 100 + 0.3 * generic / 25)


def _content(m: dict) -> float:
    verbs = 10 * min(1.0, m["strong_ratio"] / 0.6)
    numbers = 7 * min(1.0, m["quantified_ratio"] / 0.3)
    writing = 8 - min(4, len(m["weak_phrases"])) - (2 if m["first_person"] > 2 else 0) \
        - min(2, m["long_bullets"])
    return verbs + numbers + max(0, writing)


def _ats_compat(m: dict) -> float:
    pts = 15.0
    pts -= 0 if m["has_email"] else 3
    pts -= 0 if m["has_phone"] else 2
    pts -= 3 if m["has_address"] else 0           # full home address = privacy risk, no ATS benefit
    pts -= 2 if m["special_chars"] > 4 else 0     # icons / box-drawing characters confuse parsers
    pts -= 3 if m["words"] < 150 else (2 if m["words"] > 1500 else 0)
    return max(0.0, pts)


def interpret(score: float) -> str:
    if score >= 85:
        return "Excellent: a strong, ATS-ready resume"
    if score >= 70:
        return "Good: a few improvements will make it stronger"
    if score >= 55:
        return "Fair: several gaps are likely to hurt ATS ranking"
    return "Needs work: fix the high-severity issues first"


def score_resume(text: str, parsed: ParsedResume, validation: SkillValidation,
                 jd_match: Optional[JDMatch] = None) -> tuple[list[ComponentScore], dict]:
    m = collect_metrics(text, parsed)
    parts = [
        ("Formatting", _formatting(text, parsed, m), 20),
        ("Keywords", _keywords(parsed, jd_match), 25),
        ("Content quality", _content(m), 25),
        ("Skill validation", 15 * validation.validation_pct / 100, 15),
        ("ATS compatibility", _ats_compat(m), 15),
    ]
    comps = [ComponentScore(name=n, score=round(min(s, mx), 1), max_score=mx) for n, s, mx in parts]
    return comps, m


def find_strengths(m: dict, validation: SkillValidation, jd_match: Optional[JDMatch]) -> list[str]:
    out = []
    if validation.validation_pct >= 70:
        out.append("Most listed skills are backed by projects or experience")
    if m["quantified_ratio"] >= 0.3:
        out.append("Good use of numbers to show impact")
    if m["strong_ratio"] >= 0.6:
        out.append("Bullets start with strong action verbs")
    if m["has_github"]:
        out.append("GitHub link included")
    if m["has_linkedin"]:
        out.append("LinkedIn link included")
    if jd_match and jd_match.match_percentage >= 70:
        out.append("Strong keyword overlap with the job description")
    return out
