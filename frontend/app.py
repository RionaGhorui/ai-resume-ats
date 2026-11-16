"""Streamlit UI. Run with:  streamlit run frontend/app.py"""
import json
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
SEV_ICON = {"High": "🔴", "Moderate": "🟠", "Low": "🟡"}


def call_backend(file, job_description: str) -> dict:
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/v1/analyze-resume",
            files={"resume": (file.name, file.getvalue())},
            data={"job_description": job_description},
            timeout=180,
        )
    except requests.ConnectionError:
        raise RuntimeError(f"Cannot reach the API at {BACKEND_URL}. Is the backend running?")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(str(detail))
    return r.json()


def to_markdown(r: dict) -> str:
    lines = [f"# ATS Report: {r['ats_score']:.0f}/100", f"_{r['interpretation']}_", "", "## Score breakdown"]
    lines += [f"- {c['name']}: {c['score']}/{c['max_score']:.0f}" for c in r["component_scores"]]
    if r["strengths"]:
        lines += ["", "## Strengths"] + [f"- {s}" for s in r["strengths"]]
    if r.get("jd_match"):
        j = r["jd_match"]
        lines += ["", f"## Job match: {j['match_percentage']:.0f}%",
                  "Missing: " + (", ".join(j["missing_keywords"]) or "none")]
    if r["suggestions"]:
        lines += ["", "## AI suggestions"]
        for s in r["suggestions"]:
            lines.append(f"- **{s['target']}**: {s['advice']}")
            if s["example_bullet"]:
                lines.append(f"  - Example: {s['example_bullet']}")
    lines += ["", "## Issues"]
    for i in r["detailed_feedback"]:
        lines += [f"### {i['severity_level']}: {i['issue_title']}", i["explanation"], f"**Fix:** {i['how_to_fix']}"]
        lines += [f"- {a}" for a in i["action_items"]] + [""]
    return "\n".join(lines)


def show(r: dict) -> None:
    jd = r.get("jd_match")
    c1, c2, c3 = st.columns(3)
    c1.metric("ATS score", f"{r['ats_score']:.0f} / 100")
    c2.metric("Job match", f"{jd['match_percentage']:.0f}%" if jd else "n/a")
    c3.metric("Skills backed by evidence", f"{r['skill_validation']['validation_pct']:.0f}%")
    st.caption(r["interpretation"])
    for w in r["warnings"]:
        st.warning(w)
    for c in r["component_scores"]:
        st.progress(min(1.0, c["score"] / c["max_score"]), text=f"{c['name']}: {c['score']} / {c['max_score']:.0f}")

    names = ["Feedback", "Skills"] + (["Job match"] if jd else []) \
        + (["AI suggestions"] if r["suggestions"] else []) + ["Download"]
    tabs = dict(zip(names, st.tabs(names)))

    with tabs["Feedback"]:
        if r["strengths"]:
            st.success("**Strengths**\n\n" + "\n".join(f"- {s}" for s in r["strengths"]))
        for i in r["detailed_feedback"]:
            with st.expander(f"{SEV_ICON[i['severity_level']]} {i['issue_title']}"):
                st.write(i["explanation"])
                st.markdown(f"**How to fix:** {i['how_to_fix']}")
                st.markdown("\n".join(f"- {a}" for a in i["action_items"]))
                st.code(i["example_improvement"], language=None)
        if not r["detailed_feedback"]:
            st.info("No issues found.")

    with tabs["Skills"]:
        sv = r["skill_validation"]
        st.markdown("**Backed by evidence**")
        for s in sv["validated"]:
            st.markdown(f"- ✅ **{s['skill']}**: {', '.join(s['evidence'])}")
        st.markdown("**Listed but not demonstrated**")
        st.write(", ".join(sv["unvalidated"]) or "None 🎉")

    if jd:
        with tabs["Job match"]:
            st.write(f"Keyword match: **{jd['keyword_match_pct']:.0f}%**"
                     + (f"  |  Meaning match: **{jd['semantic_similarity'] * 100:.0f}%**"
                        if jd.get("semantic_similarity") is not None else ""))
            st.markdown("**Matched:** " + (", ".join(jd["matched_keywords"]) or "none"))
            st.markdown("**Missing:** " + (", ".join(jd["missing_keywords"]) or "none"))
            if jd["skills_gap"]:
                st.error("Required skills missing: " + ", ".join(jd["skills_gap"]))
            if jd.get("requirement_matches"):
                st.markdown("**Best evidence in your resume for each duty** (retrieved by meaning)")
                st.dataframe(
                    [{"Job duty": m["requirement"], "Best match in resume": m["source"], "Similarity": m["score"]}
                     for m in jd["requirement_matches"]],
                    hide_index=True, use_container_width=True)

    if r["suggestions"]:
        with tabs["AI suggestions"]:
            for s in r["suggestions"]:
                st.markdown(f"**{s['target']}**: {s['advice']}")
                if s["example_bullet"]:
                    st.code(s["example_bullet"], language=None)

    with tabs["Download"]:
        st.download_button("Markdown report", to_markdown(r), "ats_report.md", "text/markdown")
        st.download_button("Raw JSON", json.dumps(r, indent=2), "ats_report.json", "application/json")


st.set_page_config(page_title="AI Resume ATS", page_icon="📄", layout="wide")
st.title("📄 AI Resume ATS")
st.caption("Upload your resume, optionally paste a job description, get a score and specific fixes.")

left, right = st.columns(2)
resume_file = left.file_uploader("Resume (PDF or DOCX, max 5 MB)", type=["pdf", "docx"])
jd_text = right.text_area("Job description (optional)", height=170)

if st.button("Analyze", type="primary", disabled=resume_file is None):
    with st.spinner("Analyzing. This takes 10–30 seconds..."):
        try:
            st.session_state["result"] = call_backend(resume_file, jd_text)
        except RuntimeError as exc:
            st.session_state.pop("result", None)
            st.error(str(exc))

if "result" in st.session_state:
    show(st.session_state["result"])
