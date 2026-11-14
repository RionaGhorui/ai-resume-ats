from backend.feedback import find_issues
from backend.jd_matcher import match_jd
from backend.matching import contains_term, match_terms, normalize
from backend.scoring import collect_metrics, score_resume
from backend.skills import validate_skills


def test_alias_and_whole_word_matching():
    assert normalize("ReactJS") == "react"
    assert contains_term("built apis with node.js", "NodeJS")
    assert not contains_term("a good idea", "Go")          # 'go' must not match inside 'good'
    assert not contains_term("i know javascript", "Java")


def test_match_terms_uses_resume_text():
    matched, missing = match_terms(["Docker", "Kubernetes"], [], "I used docker daily")
    assert matched == ["Docker"] and missing == ["Kubernetes"]


def test_skill_validation_ignores_the_skills_list(parsed_resume):
    v = validate_skills(parsed_resume)
    names = {s.skill for s in v.validated}
    assert {"Python", "FastAPI", "Docker"} <= names
    assert "Linux" in v.unvalidated and "Postman" in v.unvalidated   # only listed, never used
    assert 0 < v.validation_pct < 100


def test_jd_match_finds_gap(resume_text, parsed_resume, parsed_jd):
    jd = match_jd(resume_text, parsed_resume, parsed_jd)
    assert "Kubernetes" in jd.skills_gap and "Python" not in jd.missing_keywords
    assert 0 < jd.match_percentage < 100


def test_scores_are_bounded_and_sensible(resume_text, parsed_resume, parsed_jd):
    v = validate_skills(parsed_resume)
    jd = match_jd(resume_text, parsed_resume, parsed_jd)
    comps, m = score_resume(resume_text, parsed_resume, v, jd)
    assert all(0 <= c.score <= c.max_score for c in comps)
    assert sum(c.max_score for c in comps) == 100
    assert 55 <= sum(c.score for c in comps) <= 95        # a decent fresher resume
    assert m["has_email"] and m["has_phone"] and m["has_github"] and not m["has_address"]
    assert "responsible for" in m["weak_phrases"]


def test_empty_resume_scores_low_and_does_not_crash():
    from backend.schemas import ParsedResume
    p = ParsedResume()
    v = validate_skills(p)
    comps, m = score_resume("hello", p, v)
    assert sum(c.score for c in comps) < 25
    titles = [i.issue_title for i in find_issues("hello", p, v, comps, m)]
    assert "Missing Projects Section" in titles and "Missing or Weak Skills Section" in titles


def test_address_detection():
    from backend.schemas import ParsedResume
    p = ParsedResume()
    assert collect_metrics("Lives at 12 MG Road, Bengaluru", p)["has_address"]
    assert collect_metrics("Bengaluru - 560001", p)["has_address"]
    assert not collect_metrics("Built 12 endpoints, 25000 users, call +91 98765 43210", p)["has_address"]


def test_feedback_includes_jd_gap(resume_text, parsed_resume, parsed_jd):
    v = validate_skills(parsed_resume)
    jd = match_jd(resume_text, parsed_resume, parsed_jd)
    comps, m = score_resume(resume_text, parsed_resume, v, jd)
    titles = [i.issue_title for i in find_issues(resume_text, parsed_resume, v, comps, m, jd)]
    assert "Required Job Skills Missing From Resume" in titles
