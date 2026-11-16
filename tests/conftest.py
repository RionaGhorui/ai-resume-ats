"""Shared fixtures. The LLM is never called in tests: we hand-write what it would return."""
from pathlib import Path

import pytest

from backend import config, models
from backend.schemas import Education, Experience, ParsedJD, ParsedResume, Project

SAMPLES = Path(__file__).resolve().parents[1] / "sample_data"

SAMPLE_TEXT = """Sample Candidate
sample.candidate@example.com | +91 90000 00000 | linkedin.com/in/sample-candidate | github.com/sample-candidate
SUMMARY
Final-year B.Tech Computer Science student who builds REST APIs and ML projects in Python.
EDUCATION
B.Tech in Computer Science and Engineering, Example Institute of Technology (2022 - 2026), CGPA: 8.4
SKILLS
Languages: Python, SQL, JavaScript
Frameworks: FastAPI, Flask, React, scikit-learn
Tools: Git, Docker, PostgreSQL, Linux, Postman, AWS, Redis
PROJECTS
Expense Tracker API - FastAPI, PostgreSQL, Docker
• Built a REST API with JWT auth and 12 endpoints used by a React dashboard
• Containerised the service with Docker and cut local setup time from 30 minutes to 5
• Wrote 40+ pytest tests, reaching 85% coverage
Movie Review Sentiment Classifier - Python, scikit-learn
• Trained a logistic regression model on 25,000 reviews achieving 88% accuracy
• Deployed a Flask demo that classifies a review in under 200 ms
EXPERIENCE
Software Engineering Intern - Example Tech Pvt Ltd (Jun 2025 - Aug 2025)
• Developed 6 internal REST endpoints in Flask used by the support team
• Reduced report generation time by 35% by adding SQL indexes
• Responsible for writing documentation
CERTIFICATIONS
AWS Cloud Practitioner (2025)
"""


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    """Every test uses the offline fake embedder, so no model is ever downloaded."""
    monkeypatch.setattr(config, "EMBEDDINGS_BACKEND", "fake")
    models.get_embeddings.cache_clear()
    yield
    models.get_embeddings.cache_clear()


@pytest.fixture
def resume_text():
    return SAMPLE_TEXT


@pytest.fixture
def parsed_resume():
    return ParsedResume(
        name="Sample Candidate", email="sample.candidate@example.com", phone="+91 90000 00000",
        linkedin="linkedin.com/in/sample-candidate", github="github.com/sample-candidate",
        professional_summary="Final-year B.Tech CS student who builds REST APIs and ML projects.",
        skills=["Python", "SQL", "JavaScript", "FastAPI", "Flask", "React", "scikit-learn",
                "Git", "Docker", "PostgreSQL", "Linux", "Postman", "AWS", "Redis"],
        education=[Education(degree="B.Tech CSE", institution="Example Institute of Technology",
                             year="2022-2026", gpa="8.4")],
        projects=[
            Project(title="Expense Tracker API", technologies=["FastAPI", "PostgreSQL", "Docker"],
                    description="Built a REST API with JWT auth and 12 endpoints used by a React dashboard\n"
                                "Containerised the service with Docker and cut local setup time from 30 minutes to 5\n"
                                "Wrote 40+ pytest tests, reaching 85% coverage"),
            Project(title="Movie Review Sentiment Classifier", technologies=["Python", "scikit-learn"],
                    description="Trained a logistic regression model on 25,000 reviews achieving 88% accuracy\n"
                                "Deployed a Flask demo that classifies a review in under 200 ms"),
        ],
        experience=[Experience(job_title="Software Engineering Intern", company="Example Tech Pvt Ltd",
                               duration_months=3,
                               description="Developed 6 internal REST endpoints in Flask used by the support team\n"
                                           "Reduced report generation time by 35% by adding SQL indexes\n"
                                           "Responsible for writing documentation")],
        certifications=["AWS Cloud Practitioner (2025)"],
        action_verbs=["Built", "Containerised", "Wrote", "Trained", "Deployed", "Developed", "Reduced"],
        keywords=["REST API", "JWT", "pytest", "Docker", "scikit-learn", "SQL", "Flask", "FastAPI"],
    )


@pytest.fixture
def parsed_jd():
    return ParsedJD(
        job_title="Backend Engineer (Fresher)",
        required_skills=["Python", "FastAPI", "Flask", "SQL", "Git", "Docker", "REST APIs", "Kubernetes"],
        preferred_skills=["AWS", "CI/CD", "pytest", "Redis"],
        keywords=["automated tests", "containers"],
        responsibilities=["Design and build REST APIs", "Write automated tests for every feature",
                          "Deploy services using containers and CI/CD pipelines",
                          "Collaborate with the frontend team on API contracts"],
    )
