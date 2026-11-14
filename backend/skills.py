"""Skill validation: is each listed skill actually used in a project / job / certificate?"""
from backend.matching import contains_term, dedupe
from backend.schemas import ParsedResume, SkillEvidence, SkillValidation


def evidence_sources(parsed: ParsedResume) -> list[tuple[str, str]]:
    """(label, text) pairs where a skill can be *demonstrated* (the skills list itself is excluded)."""
    out = []
    for p in parsed.projects:
        text = " ".join([p.title, p.description, " ".join(p.technologies)]).strip()
        if text:
            out.append((f"Project: {p.title or 'Untitled'}", text))
    for e in parsed.experience:
        text = " ".join([e.job_title, e.company, e.description]).strip()
        if text:
            label = f"Experience: {e.job_title or 'Role'}" + (f" @ {e.company}" if e.company else "")
            out.append((label, text))
    if parsed.certifications:
        out.append(("Certifications", ", ".join(parsed.certifications)))
    return out


def validate_skills(parsed: ParsedResume) -> SkillValidation:
    sources = [(label, text.lower()) for label, text in evidence_sources(parsed)]
    validated, unvalidated = [], []
    for skill in dedupe(parsed.skills):
        labels = [label for label, text in sources if contains_term(text, skill)]
        if labels:
            validated.append(SkillEvidence(skill=skill, evidence=labels[:3]))
        else:
            unvalidated.append(skill)
    total = len(validated) + len(unvalidated)
    pct = round(100 * len(validated) / total, 1) if total else 0.0
    return SkillValidation(validated=validated, unvalidated=unvalidated, validation_pct=pct)
