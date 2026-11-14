"""Turn scores + metrics into human-readable issues (rule-based, no LLM)."""
from typing import Optional

from backend.schemas import ComponentScore, IssueDetail, JDMatch, ParsedResume, SkillValidation
from backend.scoring import has_header

_ORDER = {"High": 0, "Moderate": 1, "Low": 2}


def _issue(title, severity, impact, why, where, fix, actions, example) -> IssueDetail:
    return IssueDetail(
        issue_title=title, severity_level=severity, ats_impact=impact, explanation=why,
        where_it_appears=where, how_to_fix=fix, action_items=actions, example_improvement=example,
    )


def find_issues(text: str, parsed: ParsedResume, validation: SkillValidation,
                components: list[ComponentScore], m: dict,
                jd_match: Optional[JDMatch] = None) -> list[IssueDetail]:
    out: list[IssueDetail] = []
    low = text.lower()
    comp = {c.name: c for c in components}

    if not parsed.projects and not has_header(text, "projects"):
        out.append(_issue(
            "Missing Projects Section", "High", "High",
            "Recruiters and ATS filters look for projects to see your skills used in practice. "
            "For a fresher this is usually the strongest section.",
            "No 'Projects' header or project entries found",
            "Add 2–3 projects with the tech stack, what you built, and a measurable result.",
            ["Add a PROJECTS section", "Name the tech stack for each project",
             "End each bullet with a result (users, speed, accuracy, tests)", "Link the GitHub repo"],
            "Expense Tracker API — FastAPI, PostgreSQL, Docker\n• Built 12 REST endpoints with JWT auth; 85% test coverage",
        ))

    if not parsed.experience and "intern" not in low:
        sev = "Moderate" if parsed.projects else "High"
        out.append(_issue(
            "No Internship or Work Experience", sev, "Medium",
            "No experience entries were found. Freshers can use internships, freelance work, "
            "open-source contributions or club leadership instead.",
            "No 'Experience' or 'Internships' section",
            "Add an Experience section with any real, relevant work, even if short.",
            ["List internships, freelance, open-source or club roles",
             "Use: Role — Organisation (Mon YYYY – Mon YYYY)", "Add 2–3 impact bullets per role"],
            "Software Engineering Intern — Example Tech (Jun 2025 – Aug 2025)\n• Reduced report time by 35% with SQL indexes",
        ))

    if not parsed.education and not has_header(text, "education"):
        out.append(_issue(
            "Missing Education Section", "Moderate", "Medium",
            "ATS systems expect your degree, institution and graduation year.",
            "No 'Education' header found",
            "Add degree, college, years and CGPA (if good).",
            ["Add an EDUCATION section near the top (freshers)", "Include degree, institution, years, CGPA"],
            "B.Tech in Computer Science — Example Institute (2022–2026), CGPA 8.4",
        ))

    if len(parsed.skills) < 5:
        out.append(_issue(
            "Missing or Weak Skills Section", "High", "High",
            f"Only {len(parsed.skills)} skill(s) were detected. ATS matching relies heavily on a clear skills list.",
            "Skills section",
            "Add a categorised skills section using the same names job descriptions use.",
            ["Group by Languages / Frameworks / Tools / Databases", "List 10–15 relevant skills",
             "Use exact names (e.g. 'PostgreSQL', not 'DB')"],
            "Languages: Python, SQL\nFrameworks: FastAPI, React\nTools: Git, Docker, AWS",
        ))

    total = len(validation.validated) + len(validation.unvalidated)
    if total >= 3 and validation.validation_pct < 50:
        out.append(_issue(
            "Most Skills Lack Supporting Evidence", "Moderate", "High",
            f"{len(validation.unvalidated)} of {total} listed skills are not mentioned in any project, "
            "internship or certificate. Recruiters treat unsupported skills as padding.",
            "Skills list vs. Projects/Experience",
            "Mention each skill inside a project or experience bullet, or remove it.",
            [f"Use '{s}' in a project bullet or remove it" for s in validation.unvalidated[:5]],
            "Built a REST API with FastAPI and PostgreSQL, containerised with Docker",
        ))

    if m["bullets"] >= 3 and (m["strong_ratio"] < 0.5 or m["weak_phrases"]):
        found = ", ".join(f"'{p}'" for p in m["weak_phrases"])
        why = (f"Weak phrasing found: {found}. " if found else "") + \
              f"{round(m['strong_ratio'] * 100)}% of bullets start with a strong action verb (aim for 60%+)."
        out.append(_issue(
            "Bullets Lack Strong Action Verbs", "Moderate", "Medium", why,
            "Experience and Projects bullets",
            "Start each bullet with a past-tense verb that shows what YOU did.",
            ["Replace 'Responsible for X' with 'Developed X'", "Use: Built, Designed, Implemented, Reduced, Led",
             "Avoid 'worked on', 'helped with', 'involved in'"],
            "Weak: Responsible for writing documentation\nStrong: Authored API documentation used by 5 teammates",
        ))

    if m["bullets"] >= 3 and m["quantified_ratio"] < 0.2:
        out.append(_issue(
            "No Quantifiable Achievements", "Moderate", "Medium",
            f"Only {round(m['quantified_ratio'] * 100)}% of bullets contain a number.",
            "Experience and Projects bullets",
            "Add numbers: users, speed-ups, accuracy, test coverage, dataset size.",
            ["Add at least one metric to every project", "Use %, counts or time saved", "Estimate honestly if exact data is missing"],
            "Trained a classifier on 25,000 reviews reaching 88% accuracy",
        ))

    missing_contact = [n for n, ok in [("email", m["has_email"]), ("phone", m["has_phone"])] if not ok]
    missing_links = [n for n, ok in [("LinkedIn", m["has_linkedin"]), ("GitHub", m["has_github"])] if not ok]
    if missing_contact or missing_links:
        out.append(_issue(
            "Incomplete Contact Information", "High" if missing_contact else "Moderate", "High" if missing_contact else "Medium",
            "Missing: " + ", ".join(missing_contact + missing_links) + ".",
            "Header / top of resume",
            "Put email, phone, LinkedIn and GitHub in one line at the top as plain text.",
            [f"Add your {x}" for x in missing_contact + missing_links],
            "name@email.com | +91 90000 00000 | linkedin.com/in/name | github.com/name",
        ))

    if m["has_address"]:
        out.append(_issue(
            "Full Home Address Included", "Low", "Low",
            "A street address or PIN code adds privacy risk and no ATS benefit.",
            "Contact block", "Keep only city and state.",
            ["Remove street address and PIN/ZIP code", "Keep 'City, State'"], "Bengaluru, Karnataka",
        ))

    if comp["Formatting"].score < 10:
        out.append(_issue(
            "Poor Resume Formatting", "High", "High",
            f"Formatting score is {comp['Formatting'].score}/20: standard section headers or bullets are missing.",
            "Overall structure",
            "Use one column, standard headers and consistent bullets.",
            ["Use headers: EDUCATION, SKILLS, PROJECTS, EXPERIENCE", "Avoid tables, text boxes and images",
             "Use a standard font at 10–12 pt"],
            "NAME & CONTACT → SUMMARY → EDUCATION → SKILLS → PROJECTS → EXPERIENCE",
        ))

    if not parsed.professional_summary and not has_header(text, "summary"):
        out.append(_issue(
            "Missing Professional Summary", "Low", "Low",
            "A 2–3 line summary helps recruiters understand your profile quickly.",
            "Top of resume", "Add a short, role-targeted summary.",
            ["Write 2 sentences: who you are + key skills + target role", "Keep it under 60 words"],
            "Final-year CS student who builds REST APIs and ML demos in Python, seeking a backend role.",
        ))

    if jd_match and jd_match.skills_gap:
        sev = "High" if jd_match.match_percentage < 50 else "Moderate"
        out.append(_issue(
            "Required Job Skills Missing From Resume", sev, "High",
            f"The job description asks for {', '.join(jd_match.skills_gap[:6])}, which your resume never mentions.",
            "Skills / Projects vs. job description",
            "Only add skills you genuinely have. If you have them, show them in a project or bullet.",
            [f"If true, mention '{s}' in a project bullet" for s in jd_match.skills_gap[:5]]
            + ["If you lack a skill, build a tiny project with it before applying"],
            "Deployed the API on Kubernetes (minikube) with 2 replicas",
        ))

    return sorted(out, key=lambda i: _ORDER[i.severity_level])
