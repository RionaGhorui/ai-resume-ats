"""Streamlit UI. Run with:  streamlit run frontend/app.py

Styling lives in frontend/styles.css (loaded by load_css) and .streamlit/config.toml.
"""
import hmac
import html
import json
import os
from pathlib import Path

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")      # set it to require a password (protects your API quota)
SEV_ICON = {"High": "🔴", "Moderate": "🟠", "Low": "🟡"}


# ---------------------------------------------------------------- helpers
def esc(value) -> str:
    """Escape text before it goes into custom HTML. Skill names etc. come from an LLM, so never trust them."""
    return html.escape(str(value), quote=True).replace("$", "&#36;")


def load_css() -> None:
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


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


# ---------------------------------------------------------------- custom HTML pieces
# Rules: build each block as ONE line (a blank line ends an HTML block in Markdown) and escape every value.
def hero() -> None:
    st.markdown(
        '<header class="hero"><h1>Check your resume the way an ATS reads it</h1>'
        "<p>Upload your resume and, if you have one, a job description. You get a score out of 100, "
        "the skills your projects can prove, and the fixes worth doing first.</p></header>",
        unsafe_allow_html=True,
    )


def chips(items: list[str], kind: str) -> str:
    if not items:
        return ""
    return '<ul class="chips">' + "".join(f'<li class="chip {kind}">{esc(i)}</li>' for i in items) + "</ul>"


def section(title: str) -> str:
    return f'<h3 class="sec">{esc(title)}</h3>'


def report_html(r: dict) -> str:
    score = r["ats_score"]
    tone = "good" if score >= 70 else "fair" if score >= 55 else "low"
    verdict, _, advice = r["interpretation"].partition(": ")
    advice = advice[:1].upper() + advice[1:]
    jd = r.get("jd_match")

    stats = []
    if jd:
        stats.append((f"{jd['match_percentage']:.0f}%", "match with the job"))
    stats.append((f"{r['skill_validation']['validation_pct']:.0f}%", "of your skills shown in projects"))
    stats.append((str(len(r["detailed_feedback"])), "fixes suggested"))

    bars = ""
    for c in r["component_scores"]:
        ratio = c["score"] / c["max_score"] if c["max_score"] else 0
        level = "ok" if ratio >= 0.75 else "mid" if ratio >= 0.5 else "low"
        bars += (
            f'<div class="bar {level}" role="meter" aria-label="{esc(c["name"])}" aria-valuemin="0" '
            f'aria-valuemax="{c["max_score"]:g}" aria-valuenow="{c["score"]:g}">'
            f'<div class="bar-head"><span>{esc(c["name"])}</span><b>{c["score"]:g} / {c["max_score"]:g}</b></div>'
            f'<div class="bar-track"><i style="width:{ratio * 100:.0f}%"></i></div></div>'
        )

    return (
        f'<section class="report tone-{tone}" aria-label="Your ATS result"><div class="report-grid">'
        f'<div class="verdict"><div class="ring" style="--p:{score:.1f}" role="img" '
        f'aria-label="Score {score:.0f} out of 100"><span class="num">{score:.0f}</span><span class="of">out of 100</span></div>'
        f'<div class="verdict-copy"><h2>{esc(verdict)}</h2><p>{esc(advice)}</p></div></div>'
        f'<div class="bars">{bars}</div></div>'
        '<ul class="stats">'
        + "".join(f"<li><b>{esc(v)}</b><span>{esc(label)}</span></li>" for v, label in stats)
        + "</ul></section>"
    )


# ---------------------------------------------------------------- results
def show(r: dict) -> None:
    jd = r.get("jd_match")
    st.markdown(report_html(r), unsafe_allow_html=True)
    for w in r["warnings"]:
        st.warning(w)

    names = ["Fixes", "Skills"] + (["Job match"] if jd else []) \
        + (["Rewrites"] if r["suggestions"] else []) + ["Download"]
    tabs = dict(zip(names, st.tabs(names)))

    with tabs["Fixes"]:
        if r["strengths"]:
            st.markdown(section("What is already working")
                        + '<ul class="strengths">' + "".join(f"<li>{esc(s)}</li>" for s in r["strengths"]) + "</ul>",
                        unsafe_allow_html=True)
        if r["detailed_feedback"]:
            st.markdown(section("Fix these, most important first"), unsafe_allow_html=True)
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
        proof = "".join(
            f'<li><span class="chip hit">{esc(s["skill"])}</span><span class="why">{esc("; ".join(s["evidence"]))}</span></li>'
            for s in sv["validated"])
        st.markdown(section("Skills your resume proves")
                    + (f'<ul class="proof">{proof}</ul>' if proof else '<p class="muted">None of your listed skills appear in a project or job.</p>')
                    + section("Skills you list but never show")
                    + (chips(sv["unvalidated"], "miss")
                       + '<p class="muted">Mention each one in a project or job bullet, or remove it from your skills list.</p>'
                       if sv["unvalidated"] else '<p class="muted">Every listed skill is backed by your projects.</p>'),
                    unsafe_allow_html=True)

    if jd:
        with tabs["Job match"]:
            sem = jd.get("semantic_similarity")
            summary = f"Keyword match {jd['keyword_match_pct']:.0f}%" + (f", meaning match {sem * 100:.0f}%" if sem is not None else "")
            body = f'<p class="muted">{esc(summary)}</p>'
            if jd["skills_gap"]:
                body += ('<p class="callout">The job lists these as required, and your resume never mentions them: '
                         f'<b>{esc(", ".join(jd["skills_gap"]))}</b>.</p>')
            body += section("Found in your resume") + (chips(jd["matched_keywords"], "hit") or '<p class="muted">None.</p>')
            body += section("Not found") + (chips(jd["missing_keywords"], "miss") or '<p class="muted">Nothing missing.</p>')
            if jd.get("requirement_matches"):
                rows = "".join(
                    f'<tr><td>{esc(m["requirement"])}</td><td>{esc(m["source"])}</td><td class="sim">{m["score"]:.2f}</td></tr>'
                    for m in jd["requirement_matches"])
                body += (section("Best evidence for each duty in the job post")
                         + '<div class="evidence-wrap"><table class="evidence"><thead><tr><th>Duty</th><th>Closest part of your resume</th>'
                           f'<th>Similarity</th></tr></thead><tbody>{rows}</tbody></table></div>')
            st.markdown(body, unsafe_allow_html=True)

    if r["suggestions"]:
        with tabs["Rewrites"]:
            for s in r["suggestions"]:
                st.markdown(section(s["target"]) + f"<p>{esc(s['advice'])}</p>", unsafe_allow_html=True)
                if s["example_bullet"]:
                    st.code(s["example_bullet"], language=None)

    with tabs["Download"]:
        st.download_button("Download report (Markdown)", to_markdown(r), "ats_report.md", "text/markdown")
        st.download_button("Download raw data (JSON)", json.dumps(r, indent=2), "ats_report.json", "application/json")


# ---------------------------------------------------------------- page
st.set_page_config(page_title="AI Resume ATS", page_icon="📄", layout="wide")
load_css()

if APP_PASSWORD and not st.session_state.get("authed"):
    hero()
    entered = st.text_input("Password", type="password")
    if entered and hmac.compare_digest(entered.encode(), APP_PASSWORD.encode()):
        st.session_state["authed"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()

hero()
left, right = st.columns(2, gap="large")
resume_file = left.file_uploader("Your resume (PDF or DOCX, up to 5 MB)", type=["pdf", "docx"])
jd_text = right.text_area("Job description (optional)", height=170,
                          placeholder="Paste the job post to see which required skills your resume is missing.")

if st.button("Check my resume", type="primary", disabled=resume_file is None):
    with st.spinner("Checking your resume. This takes 10 to 30 seconds."):
        try:
            st.session_state["result"] = call_backend(resume_file, jd_text)
        except RuntimeError as exc:
            st.session_state.pop("result", None)
            st.error(str(exc))

if "result" in st.session_state:
    show(st.session_state["result"])
