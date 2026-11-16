"""Skill validation: is each listed skill actually used in a project / job / certificate?"""
from backend import config
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


def validate_skills(parsed: ParsedResume, index=None) -> SkillValidation:
    """Exact (alias-aware) match first. If `index` is given, fall back to RAG retrieval by meaning."""
    from backend.rag import retrieve    # local import: Phase 1/2 style callers need no embeddings

    sources = [(label, text.lower()) for label, text in evidence_sources(parsed)]
    validated, unvalidated = [], []
    for skill in dedupe(parsed.skills):
        labels = [label for label, text in sources if contains_term(text, skill)]
        if labels:
            validated.append(SkillEvidence(skill=skill, evidence=labels[:3]))
            continue
        hit = retrieve(index, f"experience using {skill}", k=1) if index is not None else []
        if hit and hit[0].score >= config.SEMANTIC_THRESHOLD:
            validated.append(SkillEvidence(
                skill=skill, evidence=[f"{hit[0].source} (meaning match {hit[0].score:.2f})"]))
        else:
            unvalidated.append(skill)
    total = len(validated) + len(unvalidated)
    pct = round(100 * len(validated) / total, 1) if total else 0.0
    return SkillValidation(validated=validated, unvalidated=unvalidated, validation_pct=pct)
