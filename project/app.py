"""
Streamlit frontend for the Resume Screening and Job Matching System.

Run with:
    streamlit run project/app.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from candidate_summary import build_candidate_summary
from inference import ModelManager
from reporting import (
    build_single_match_report,
    rankings_to_csv_bytes,
    report_to_json_bytes,
    report_to_markdown,
    report_to_pdf_bytes,
)
from resume_parser import parse_resume_profile
from semantic_matching import semantic_similarity
from skill_extraction import SkillExtractor
from text_extraction import TextExtractionError, extract_text_from_upload


SAMPLE_RESUME = """Senior Machine Learning Engineer

Python developer with 5 years of experience building production ML systems.
Skills: Python, FastAPI, Docker, AWS, SQL, Pandas, NumPy, scikit-learn,
Machine Learning, data pipelines, model deployment, communication, mentoring.

Experience:
- Built FastAPI services for model inference and monitoring.
- Deployed ML workflows on AWS using Docker and CI/CD.
- Partnered with product teams to explain model decisions and improve adoption.
"""

SAMPLE_JOB = """Machine Learning Engineer

We are hiring a machine learning engineer to build production AI systems.
Required skills include Python, FastAPI, Docker, AWS, SQL, Machine Learning,
model evaluation, data pipelines, communication, and collaboration.

Preferred qualifications:
- Experience deploying models to production.
- Familiarity with Kubernetes and MLOps practices.
- Ability to explain model outputs to non-technical stakeholders.
"""

SAMPLE_BATCH = [
    (
        "Candidate A - ML Engineer",
        """Python ML engineer with FastAPI, Docker, AWS, SQL, scikit-learn,
        Pandas, model deployment, and communication experience.""",
    ),
    (
        "Candidate B - Frontend Developer",
        """React and TypeScript developer with Vue.js, design systems,
        Tailwind CSS, accessibility, and frontend testing experience.""",
    ),
    (
        "Candidate C - Data Engineer",
        """Data engineer with SQL, Spark, AWS, Docker, Airflow, PostgreSQL,
        data warehousing, and stakeholder communication experience.""",
    ),
]


st.set_page_config(
    page_title="TalentMatch AI",
    page_icon="TM",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1240px; padding-top: 1.1rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: 0; }
    h1 { font-size: 2.25rem; margin-bottom: 0.2rem; }
    h2 { font-size: 1.45rem; }
    h3 { font-size: 1.1rem; }
    div[data-testid="stMetric"] {
        border: 1px solid #e7eaf0;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
    }
    .status-box {
        border: 1px solid #d8dee9;
        border-radius: 8px;
        padding: 14px 16px;
        background: #f8fafc;
        margin: 0.5rem 0 1rem;
    }
    .hero-strip {
        border: 1px solid #dbe3ef;
        border-radius: 8px;
        background: #f8fbff;
        padding: 16px 18px;
        margin: 0.75rem 0 1rem;
    }
    .section-band {
        border-top: 1px solid #e5e9f0;
        padding-top: 1rem;
        margin-top: 1rem;
    }
    .candidate-card {
        border: 1px solid #e0e6ef;
        border-radius: 8px;
        padding: 16px;
        background: #ffffff;
        margin: 0.75rem 0;
    }
    .candidate-card strong { font-size: 1.02rem; }
    .pill-row { margin-top: 0.35rem; }
    .pill {
        display: inline-block;
        border: 1px solid #cbd5e1;
        border-radius: 999px;
        padding: 2px 9px;
        margin: 2px 4px 2px 0;
        font-size: 0.86rem;
        background: #f8fafc;
        color: #0f172a;
    }
    .pill.good { border-color: #8cc7a3; background: #f0fdf4; color: #14532d; }
    .pill.warn { border-color: #e2bd68; background: #fffbeb; color: #78350f; }
    .pill.bad { border-color: #e9a0a0; background: #fef2f2; color: #7f1d1d; }
    .decision {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.84rem;
        font-weight: 700;
    }
    .decision.shortlist { background: #dcfce7; color: #14532d; }
    .decision.review { background: #fef3c7; color: #78350f; }
    .decision.reject { background: #fee2e2; color: #7f1d1d; }
    .strong { border-left: 6px solid #16803c; }
    .review { border-left: 6px solid #b7791f; }
    .weak { border-left: 6px solid #9b2c2c; }
    .small-muted { color: #64748b; font-size: 0.9rem; }
    .muted-block {
        color: #475569;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    manager = ModelManager(models_dir="./models")
    return manager.load_model("latest")


@st.cache_resource
def load_skill_extractor():
    return SkillExtractor()


def main() -> None:
    model = load_model()
    skill_extractor = load_skill_extractor()
    model_info = model.get_model_info()

    with st.sidebar:
        st.title("TalentMatch AI")
        st.caption("Recruiter workspace")
        screening_mode = st.selectbox(
            "Screening mode",
            ["Balanced", "Strict", "Broad search"],
            index=0,
            help="Balanced is best for demos. Strict favors must-have coverage. Broad search keeps more candidates in review.",
        )
        threshold = screening_threshold(screening_mode)
        st.write(f"Decision threshold: `{threshold:.0%}`")
        with st.expander("Advanced evidence settings"):
            top_k = st.slider("Evidence keywords", 3, 15, 8, 1)
        st.divider()
        render_model_status(model_info)

    st.title("TalentMatch AI")
    st.caption("Build an evidence-backed shortlist from one job description and many resumes.")
    st.markdown(
        """
        <div class="hero-strip">
            <b>Recruiter workflow:</b> define the role, upload candidate resumes, review ranked evidence,
            and export a shortlist your hiring manager can act on.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "Shortlist Builder",
        "Candidate Deep Dive",
        "Skill Scorecard",
        "Model & Trust",
        "Portfolio",
    ])

    with tabs[0]:
        batch_ranking_tab(model, skill_extractor, threshold, top_k, screening_mode)

    with tabs[1]:
        single_match_tab(model, skill_extractor, threshold, top_k)

    with tabs[2]:
        skill_analysis_tab(skill_extractor)

    with tabs[3]:
        transparency_tab(model_info)

    with tabs[4]:
        about_tab()


def single_match_tab(model, skill_extractor: SkillExtractor, threshold: float, top_k: int) -> None:
    st.subheader("Review one candidate in depth")
    st.caption("Use this when a hiring manager asks why one candidate should move forward or be held back.")

    actions = st.columns([1, 1, 5])
    with actions[0]:
        if st.button("Load sample", use_container_width=True):
            st.session_state["single_resume"] = SAMPLE_RESUME
            st.session_state["single_job"] = SAMPLE_JOB
    with actions[1]:
        if st.button("Clear", use_container_width=True):
            st.session_state["single_resume"] = ""
            st.session_state["single_job"] = ""

    upload_col, paste_col = st.columns([1, 2])
    with upload_col:
        st.markdown("#### Upload resume")
        uploaded_resume = st.file_uploader("TXT, PDF, or DOCX", type=["txt", "pdf", "docx"], key="single_upload")
        if uploaded_resume is not None:
            try:
                st.session_state["single_resume"] = extract_text_from_upload(uploaded_resume, uploaded_resume.name)
                st.success(f"Loaded {uploaded_resume.name}")
            except TextExtractionError as exc:
                st.error(str(exc))

        st.markdown("#### Result guide")
        st.write("Shortlist: strong score and clear skill evidence")
        st.write("Review: useful evidence with gaps to validate")
        st.write("Do not advance: low fit or missing must-have signals")

    with paste_col:
        col1, col2 = st.columns(2)
        with col1:
            resume_text = st.text_area(
                "Resume text",
                height=300,
                key="single_resume",
                placeholder="Paste a candidate resume or upload one from the left.",
            )
        with col2:
            job_text = st.text_area(
                "Job description",
                height=300,
                key="single_job",
                placeholder="Paste the job description here.",
            )

    if st.button("Build candidate brief", type="primary", use_container_width=True):
        if not resume_text.strip() or not job_text.strip():
            st.error("Please provide both resume text and job description.")
            return
        render_single_result(model, skill_extractor, resume_text, job_text, threshold, top_k)


def render_single_result(
    model,
    skill_extractor: SkillExtractor,
    resume_text: str,
    job_text: str,
    threshold: float,
    top_k: int,
) -> None:
    with st.spinner("Analyzing fit, skills, and explanation..."):
        started = time.time()
        prediction = model.predict_with_explanation(resume_text, job_text, top_k_keywords=top_k, threshold=threshold)
        skill_match = skill_extractor.match_skills(resume_text, job_text)
        profile = parse_resume_profile(resume_text, skill_match["matched_skills"])
        semantic = semantic_similarity(resume_text, job_text)
        criteria = infer_job_criteria(skill_extractor, job_text)
        elapsed_ms = (time.time() - started) * 1000

    score = float(prediction["match_score"])
    scorecard = build_candidate_scorecard(profile, skill_match, criteria, score, threshold)
    recommendation = scorecard["recommendation"]
    summary = build_candidate_summary(
        profile,
        score,
        recommendation,
        skill_match["matched_skills"],
        skill_match["missing_skills"],
    )
    status_class = "strong" if score >= 0.8 else "review" if score >= threshold else "weak"

    st.markdown("---")
    st.markdown(f"<div class='status-box {status_class}'><b>{recommendation}</b><br>"
                f"<span class='small-muted'>Generated in {elapsed_ms:.1f} ms</span></div>",
                unsafe_allow_html=True)

    metrics = st.columns(5)
    metrics[0].metric("Match score", f"{score:.1%}")
    metrics[1].metric("Decision", "Match" if prediction["is_match"] else "Review")
    metrics[2].metric("Semantic signal", f"{semantic['score']:.1%}")
    metrics[3].metric("Matched skills", len(skill_match["matched_skills"]))
    metrics[4].metric("Missing skills", len(skill_match["missing_skills"]))

    profile_col, summary_col = st.columns([1, 2])
    with profile_col:
        st.markdown("#### Parsed profile")
        st.write(f"Name: `{profile['candidate_name']}`")
        st.write(f"Email: `{profile['email'] or 'Not found'}`")
        st.write(f"Phone: `{profile['phone'] or 'Not found'}`")
        st.write(f"Experience: `{profile['years_experience'] or 'Not found'} years`")
        if profile["sections"]:
            st.caption("Detected sections: " + ", ".join(profile["sections"]))
    with summary_col:
        st.markdown("#### Recruiter brief")
        st.info(summary)
        st.caption(f"Semantic method: {semantic['method']}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Must-have evidence")
        render_inline_pills(scorecard["must_have_matched"], "good")
    with right:
        st.markdown("#### Must-have gaps")
        render_inline_pills(scorecard["must_have_missing"], "bad")

    st.markdown("#### Why this recommendation?")
    positives = skill_match["matched_skills"][:8]
    gaps = skill_match["missing_skills"][:8]
    if positives:
        st.success("Positive signals: " + ", ".join(positives))
    if gaps:
        st.warning("Skill gaps: " + ", ".join(gaps))
    st.info(prediction.get("explanation", "No explanation available."))
    st.caption(
        "The score combines the trained classifier with explicit skill overlap; "
        "the semantic signal is shown as additional review evidence."
    )

    st.markdown("#### Interview focus")
    for question in scorecard["interview_questions"]:
        st.write(f"- {question}")

    if prediction.get("top_keywords"):
        st.markdown("#### Top shared keywords")
        for keyword, weight in prediction["top_keywords"][:top_k]:
            st.progress(min(1.0, float(weight)), text=f"{keyword} ({float(weight):.2f})")

    report = build_single_match_report(
        resume_name="Candidate",
        job_name="Target role",
        prediction=prediction,
        skill_match=skill_match,
        recommendation=recommendation,
    )
    report["candidate_profile"] = profile
    report["candidate_summary"] = summary
    report["semantic_similarity"] = semantic
    report["scorecard"] = scorecard
    downloads = st.columns(3)
    downloads[0].download_button(
        "Download JSON",
        data=report_to_json_bytes(report),
        file_name="candidate_match_report.json",
        mime="application/json",
        use_container_width=True,
    )
    downloads[1].download_button(
        "Download Markdown",
        data=report_to_markdown(report),
        file_name="candidate_match_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    downloads[2].download_button(
        "Download PDF",
        data=report_to_pdf_bytes(report),
        file_name="candidate_match_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def batch_ranking_tab(
    model,
    skill_extractor: SkillExtractor,
    threshold: float,
    top_k: int,
    screening_mode: str,
) -> None:
    st.subheader("Build a ranked shortlist")

    if st.button("Load recruiter demo", type="secondary", use_container_width=True):
        st.session_state["batch_job"] = SAMPLE_JOB
        st.session_state["manual_batch"] = "\n---\n".join(text for _, text in SAMPLE_BATCH)

    setup_col, intake_col = st.columns([1.05, 1])
    with setup_col:
        st.markdown("#### Role intake")
        job_text = st.text_area(
            "Job description",
            height=250,
            key="batch_job",
            placeholder="Paste the role description. Include required skills, nice-to-haves, seniority, and responsibilities.",
        )
        criteria = infer_job_criteria(skill_extractor, job_text)
        if job_text.strip():
            render_role_brief(criteria)

    with intake_col:
        st.markdown("#### Candidate intake")
        uploaded_files = st.file_uploader(
            "Upload resumes",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            key="batch_uploads",
            help="Upload several resumes at once. TXT, DOCX, and selectable-text PDF are supported.",
        )
        manual_batch = st.text_area(
            "Paste resumes",
            height=174,
            key="manual_batch",
            placeholder="Optional. Separate candidates with a line containing ---",
        )
        st.caption("PDF failed? Try a DOCX export or paste the resume text here.")

    if st.button("Generate shortlist", type="primary", use_container_width=True):
        candidates = collect_batch_candidates(uploaded_files, manual_batch)
        if not job_text.strip():
            st.error("Please provide a job description.")
            return
        if not candidates:
            st.error("Please upload or paste at least one resume.")
            return

        rankings = rank_candidates(model, skill_extractor, candidates, job_text, threshold, top_k, criteria)
        render_rankings(rankings, criteria, screening_mode)


def collect_batch_candidates(uploaded_files, manual_batch: str) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    for uploaded in uploaded_files or []:
        try:
            candidates.append((uploaded.name, extract_text_from_upload(uploaded, uploaded.name)))
        except TextExtractionError as exc:
            st.warning(f"{uploaded.name}: {exc}")

    for index, block in enumerate(manual_batch.split("\n---\n"), start=1):
        cleaned = block.strip()
        if cleaned:
            candidates.append((f"Pasted candidate {index}", cleaned))
    return candidates


def rank_candidates(
    model,
    skill_extractor: SkillExtractor,
    candidates: List[Tuple[str, str]],
    job_text: str,
    threshold: float,
    top_k: int,
    criteria: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    criteria = criteria or infer_job_criteria(skill_extractor, job_text)
    with st.spinner(f"Ranking {len(candidates)} candidate(s)..."):
        for name, resume_text in candidates:
            prediction = model.predict_with_explanation(resume_text, job_text, top_k_keywords=top_k, threshold=threshold)
            skill_match = skill_extractor.match_skills(resume_text, job_text)
            profile = parse_resume_profile(resume_text, skill_match["matched_skills"])
            semantic = semantic_similarity(resume_text, job_text)
            score = float(prediction["match_score"])
            scorecard = build_candidate_scorecard(profile, skill_match, criteria, score, threshold)
            recommendation = scorecard["recommendation"]
            rows.append({
                "candidate": name,
                "match_score": round(score, 4),
                "semantic_score": round(float(semantic["score"]), 4),
                "semantic_method": semantic["method"],
                "recommendation": recommendation,
                "matched_skills": skill_match["matched_skills"],
                "missing_skills": skill_match["missing_skills"],
                "extra_skills": skill_match.get("extra_skills", []),
                "inference_time_ms": round(float(prediction["inference_time_ms"]), 2),
                "explanation": prediction.get("explanation", ""),
                "profile": profile,
                "summary": build_candidate_summary(
                    profile,
                    score,
                    recommendation,
                    skill_match["matched_skills"],
                    skill_match["missing_skills"],
                ),
                "scorecard": scorecard,
            })

    rows.sort(key=lambda item: item["match_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def render_rankings(rankings: List[Dict[str, Any]], criteria: Dict[str, Any], screening_mode: str) -> None:
    st.markdown("#### Shortlist results")
    shortlisted = [row for row in rankings if row["scorecard"]["decision_group"] == "shortlist"]
    review = [row for row in rankings if row["scorecard"]["decision_group"] == "review"]
    not_recommended = [row for row in rankings if row["scorecard"]["decision_group"] == "reject"]

    metrics = st.columns(4)
    metrics[0].metric("Candidates reviewed", len(rankings))
    metrics[1].metric("Shortlist", len(shortlisted))
    metrics[2].metric("Needs review", len(review))
    metrics[3].metric("Mode", screening_mode)

    if rankings:
        best = rankings[0]
        st.markdown(
            f"""
            <div class="status-box strong">
                <b>Top recommendation: {best['candidate']}</b><br>
                <span class="small-muted">{best['summary']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    table_rows = [
        {
            "Rank": row["rank"],
            "Candidate": row["candidate"],
            "Score": f"{row['match_score']:.1%}",
            "Decision": row["recommendation"],
            "Must-have coverage": f"{row['scorecard']['must_have_coverage']:.0%}",
            "Matched": len(row["scorecard"]["must_have_matched"]),
            "Gaps": len(row["scorecard"]["must_have_missing"]),
        }
        for row in rankings
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    for row in rankings:
        render_candidate_card(row)

    st.markdown("#### Hiring manager handoff")
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download shortlist CSV",
        data=rankings_to_csv_bytes(rankings),
        file_name="talentmatch_shortlist.csv",
        mime="text/csv",
        use_container_width=True,
    )
    downloads[1].download_button(
        "Download shortlist brief",
        data=shortlist_to_markdown(rankings, criteria),
        file_name="talentmatch_shortlist_brief.md",
        mime="text/markdown",
        use_container_width=True,
    )

    if not_recommended:
        st.caption(
            f"{len(not_recommended)} candidate(s) were not recommended because the fit score or must-have coverage was too low."
        )


def screening_threshold(mode: str) -> float:
    if mode == "Strict":
        return 0.68
    if mode == "Broad search":
        return 0.42
    return 0.55


def infer_job_criteria(skill_extractor: SkillExtractor, job_text: str) -> Dict[str, Any]:
    """Build a recruiter-friendly scorecard from the job description."""
    if not job_text.strip():
        return {
            "title": "Target role",
            "must_have": [],
            "nice_to_have": [],
            "soft_skills": [],
            "experience": None,
        }

    extracted = skill_extractor.extract_from_text(job_text)
    known_hard = set(skill_extractor.hard_skills_lower)
    known_soft = set(skill_extractor.soft_skills_lower)
    hard_skills = _skill_names(extracted["hard_skills"], known_hard)
    soft_skills = [skill for skill in _skill_names(extracted["soft_skills"], known_soft) if skill not in hard_skills]

    required_text = _lines_matching(
        job_text,
        ["required", "must", "need", "qualification", "responsibilities", "requirements"],
    )
    preferred_text = _lines_matching(
        job_text,
        ["preferred", "nice", "plus", "bonus", "familiarity", "advantage"],
    )

    required_skills = (
        _skill_names(skill_extractor.extract_from_text(required_text)["hard_skills"], known_hard)
        if required_text else []
    )
    preferred_skills = (
        _skill_names(skill_extractor.extract_from_text(preferred_text)["hard_skills"], known_hard)
        if preferred_text else []
    )

    must_have = required_skills or hard_skills[:8]
    nice_to_have = [skill for skill in (preferred_skills or hard_skills[8:14]) if skill not in must_have]

    return {
        "title": _guess_job_title(job_text),
        "must_have": must_have,
        "nice_to_have": nice_to_have,
        "soft_skills": soft_skills[:5],
        "experience": _find_required_experience(job_text),
    }


def render_role_brief(criteria: Dict[str, Any]) -> None:
    st.markdown("#### Screening scorecard")
    st.write(f"Role: `{criteria['title']}`")
    if criteria.get("experience"):
        st.write(f"Experience signal: `{criteria['experience']}+ years`")
    st.markdown("Must-have skills")
    render_inline_pills(criteria.get("must_have", []), "good")
    st.markdown("Nice-to-have skills")
    render_inline_pills(criteria.get("nice_to_have", []), "warn")
    st.markdown("Collaboration signals")
    render_inline_pills(criteria.get("soft_skills", []), "good")


def build_candidate_scorecard(
    profile: Dict[str, Any],
    skill_match: Dict[str, Any],
    criteria: Dict[str, Any],
    score: float,
    threshold: float,
) -> Dict[str, Any]:
    matched = set(skill_match.get("matched_skills", []))
    missing = set(skill_match.get("missing_skills", []))
    must_have = criteria.get("must_have", [])
    nice_to_have = criteria.get("nice_to_have", [])

    must_matched = [skill for skill in must_have if skill in matched]
    must_missing = [skill for skill in must_have if skill in missing or skill not in matched]
    nice_matched = [skill for skill in nice_to_have if skill in matched]
    nice_missing = [skill for skill in nice_to_have if skill not in matched]
    coverage = len(must_matched) / len(must_have) if must_have else 0.0

    if score >= max(0.78, threshold + 0.12) and coverage >= 0.6:
        group = "shortlist"
        recommendation = "Shortlist"
    elif score >= threshold or coverage >= 0.45:
        group = "review"
        recommendation = "Review"
    else:
        group = "reject"
        recommendation = "Do not advance"

    concerns = []
    if must_missing:
        concerns.append("Missing must-have evidence: " + ", ".join(must_missing[:4]))
    if criteria.get("experience") and not profile.get("years_experience"):
        concerns.append("Years of experience not clearly stated")
    if not concerns:
        concerns.append("No major screening concern detected from the parsed resume")

    return {
        "decision_group": group,
        "recommendation": recommendation,
        "must_have_coverage": coverage,
        "must_have_matched": must_matched,
        "must_have_missing": must_missing,
        "nice_to_have_matched": nice_matched,
        "nice_to_have_missing": nice_missing,
        "concerns": concerns,
        "interview_questions": build_interview_questions(must_missing, nice_missing, criteria),
    }


def render_candidate_card(row: Dict[str, Any]) -> None:
    scorecard = row["scorecard"]
    decision_class = {
        "shortlist": "shortlist",
        "review": "review",
        "reject": "reject",
    }[scorecard["decision_group"]]

    st.markdown(
        f"""
        <div class="candidate-card">
            <strong>#{row['rank']} {row['candidate']}</strong>
            <span class="decision {decision_class}">{scorecard['recommendation']}</span><br>
            <span class="small-muted">Fit score {row['match_score']:.1%} | Must-have coverage {scorecard['must_have_coverage']:.0%}</span>
            <p class="muted-block">{row['summary']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Evidence for {row['candidate']}", expanded=row["rank"] <= 2):
        profile = row["profile"]
        profile_cols = st.columns(4)
        profile_cols[0].write(f"Name: `{profile['candidate_name']}`")
        profile_cols[1].write(f"Email: `{profile['email'] or 'Not found'}`")
        profile_cols[2].write(f"Phone: `{profile['phone'] or 'Not found'}`")
        profile_cols[3].write(f"Experience: `{profile['years_experience'] or 'Not found'}`")

        st.markdown("Must-have evidence")
        render_inline_pills(scorecard["must_have_matched"], "good")
        if scorecard["must_have_missing"]:
            render_inline_pills(scorecard["must_have_missing"], "bad")

        st.markdown("Nice-to-have evidence")
        render_inline_pills(scorecard["nice_to_have_matched"], "good")
        if scorecard["nice_to_have_missing"]:
            render_inline_pills(scorecard["nice_to_have_missing"], "warn")

        st.markdown("Concerns")
        for concern in scorecard["concerns"]:
            st.warning(concern)

        st.markdown("Interview focus")
        for question in scorecard["interview_questions"]:
            st.write(f"- {question}")

        st.caption(f"Semantic method: {row['semantic_method']}. Model evidence: {row['explanation']}")


def render_inline_pills(items: List[str], kind: str) -> None:
    if not items:
        st.caption("None detected")
        return
    html = "<div class='pill-row'>" + "".join(
        f"<span class='pill {kind}'>{item}</span>" for item in items[:14]
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def build_interview_questions(
    must_missing: List[str],
    nice_missing: List[str],
    criteria: Dict[str, Any],
) -> List[str]:
    questions: List[str] = []
    for skill in must_missing[:3]:
        questions.append(f"Can you describe a recent project where you used {skill}?")
    for skill in nice_missing[:2]:
        questions.append(f"Have you worked with {skill}, and how quickly could you ramp up if needed?")
    if not questions:
        title = criteria.get("title", "this role")
        questions.append(f"What accomplishment best shows your readiness for {title}?")
        questions.append("Which part of this role would require the fastest ramp-up for you?")
    return questions[:4]


def shortlist_to_markdown(rankings: List[Dict[str, Any]], criteria: Dict[str, Any]) -> str:
    lines = [
        "# TalentMatch Shortlist Brief",
        "",
        f"Role: {criteria.get('title', 'Target role')}",
        f"Must-have skills: {_format_list(criteria.get('must_have', []))}",
        f"Nice-to-have skills: {_format_list(criteria.get('nice_to_have', []))}",
        "",
        "## Ranked Candidates",
    ]
    for row in rankings:
        scorecard = row["scorecard"]
        lines.extend([
            "",
            f"### #{row['rank']} {row['candidate']} - {scorecard['recommendation']}",
            f"Fit score: {row['match_score']:.1%}",
            f"Must-have coverage: {scorecard['must_have_coverage']:.0%}",
            f"Summary: {row['summary']}",
            f"Matched must-haves: {_format_list(scorecard['must_have_matched'])}",
            f"Missing must-haves: {_format_list(scorecard['must_have_missing'])}",
            f"Interview focus: {'; '.join(scorecard['interview_questions'])}",
        ])
    return "\n".join(lines).encode("utf-8")


def _skill_names(skills: List[Dict[str, Any]], allowed: Optional[set] = None) -> List[str]:
    seen = set()
    names = []
    for skill in skills:
        name = str(skill.get("name", "")).strip().lower()
        if name and name not in seen and (allowed is None or name in allowed):
            names.append(name)
            seen.add(name)
    return names


def _lines_matching(text: str, keywords: List[str]) -> str:
    lines = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            lines.append(line)
    return "\n".join(lines)


def _guess_job_title(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip(" -:\t")
        if 4 <= len(cleaned) <= 80 and not cleaned.lower().startswith(("we are", "required", "preferred")):
            return cleaned
    return "Target role"


def _find_required_experience(text: str) -> Optional[int]:
    matches = re.findall(r"(\d{1,2})\+?\s*(?:years|yrs)", text, flags=re.I)
    years = [int(value) for value in matches if int(value) < 50]
    return min(years) if years else None


def _format_list(items: List[str]) -> str:
    return ", ".join(items) if items else "None detected"


def skill_analysis_tab(skill_extractor: SkillExtractor) -> None:
    st.subheader("Inspect a role or resume scorecard")
    st.caption("Use this to sanity-check the skills the system will use as screening evidence.")
    uploaded = st.file_uploader("Upload a resume or job file", type=["txt", "pdf", "docx"], key="skill_file")
    text = st.text_area("Or paste text", height=260, key="skill_text")

    if uploaded is not None:
        try:
            text = extract_text_from_upload(uploaded, uploaded.name)
            st.success(f"Loaded {uploaded.name}")
        except TextExtractionError as exc:
            st.error(str(exc))

    if st.button("Extract scorecard signals", type="primary"):
        if not text.strip():
            st.error("Please provide text to analyze.")
            return
        result = skill_extractor.extract_from_text(text)
        criteria = infer_job_criteria(skill_extractor, text)
        render_role_brief(criteria)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Technical skills")
            render_skill_chips([item["name"] for item in result["hard_skills"]], "success")
        with col2:
            st.markdown("#### Soft skills")
            render_skill_chips([item["name"] for item in result["soft_skills"]], "info")


def transparency_tab(model_info: Dict[str, Any]) -> None:
    st.subheader("Model, trust, and responsible use")
    render_model_status(model_info)

    if model_info.get("model_version") == "demo-fallback":
        st.warning(
            "The app is currently using the demo fallback because no trained pipeline.joblib was found. "
            "It is still useful for UX demos, but train the sklearn pipeline before claiming evaluated ML metrics."
        )
    else:
        st.success("A saved trained model artifact is loaded.")

    st.markdown("#### How the recommendation works")
    st.write(
        "TalentMatch combines the trained resume-job classifier with explicit skill coverage. "
        "The UI shows the evidence behind each recommendation so a recruiter can validate the shortlist."
    )

    st.markdown("#### Recruiter guardrails")
    st.write(
        "- Treat the result as first-pass decision support, not an automatic rejection system."
    )
    st.write("- Validate missing must-haves during recruiter screen or hiring manager review.")
    st.write("- Keep a consistent scorecard for all candidates in the same role.")
    st.write("- Review model performance and bias before any production deployment.")

    st.markdown("#### What to improve before production")
    st.write("- Train on labeled resume-job pairs and save the model artifact.")
    st.write("- Validate precision, recall, F1, ROC-AUC, and subgroup fairness.")
    st.write("- Add recruiter feedback loops and audit logs.")


def about_tab() -> None:
    st.subheader("Portfolio")
    st.markdown("#### AI-Powered Resume Screening and Job Matching System")
    st.write(
        "Built by Abdul Ahad as a recruiter-focused AI screening workflow. "
        "The app helps hiring teams turn a job description and multiple resumes into a ranked, explainable shortlist."
    )

    contact_cols = st.columns(2)
    contact_cols[0].write("Email: `abdulahad.de@outlook.com`")
    contact_cols[1].write("GitHub: `aahad699`")

    st.markdown("#### Problem")
    st.write(
        "Recruiters often screen many resumes manually, compare inconsistent evidence, and spend too much time "
        "turning raw candidate documents into a defensible shortlist."
    )

    st.markdown("#### Solution")
    st.write(
        "TalentMatch AI extracts role requirements, separates must-have and nice-to-have skills, ranks candidates, "
        "flags gaps, suggests interview focus areas, and exports a recruiter-ready shortlist report."
    )

    st.markdown("#### What the project demonstrates")
    st.write("- End-to-end AI product workflow with Streamlit and FastAPI")
    st.write("- Resume ingestion for TXT, PDF, and DOCX files")
    st.write("- TF-IDF + Logistic Regression model with explicit skill-overlap evidence")
    st.write("- Recruiter scorecards, candidate cards, concerns, and interview prompts")
    st.write("- CSV, Markdown, JSON, and PDF export support")


def render_model_status(model_info: Dict[str, Any]) -> None:
    version = model_info.get("model_version", "unknown")
    model_type = model_info.get("model_type", "unknown")
    metrics = model_info.get("metrics") or {}
    st.markdown("#### Model status")
    st.write(f"Version: `{version}`")
    st.write(f"Type: `{model_type}`")
    if model_info.get("training_date"):
        st.write(f"Training date: `{model_info['training_date']}`")
    if model_info.get("feature_count"):
        st.write(f"Features: `{model_info['feature_count']}`")
    if model_info.get("vocabulary_size"):
        st.write(f"Vocabulary: `{model_info['vocabulary_size']}`")
    if metrics and version != "demo-fallback":
        metric_cols = st.columns(4)
        metric_cols[0].metric("Precision", f"{float(metrics.get('precision', 0)):.1%}")
        metric_cols[1].metric("Recall", f"{float(metrics.get('recall', 0)):.1%}")
        metric_cols[2].metric("F1", f"{float(metrics.get('f1_score', 0)):.1%}")
        metric_cols[3].metric("ROC-AUC", f"{float(metrics.get('roc_auc', 0)):.1%}")
        if metrics.get("note"):
            st.caption(metrics["note"])
    if model_info.get("training_samples") or model_info.get("test_samples"):
        st.caption(
            f"Evaluation data: {model_info.get('training_samples', 'unknown')} training pairs, "
            f"{model_info.get('test_samples', 'unknown')} test pairs."
        )
    st.caption(
        "Portfolio note: current artifacts are suitable for demonstration. Replace synthetic data with "
        "representative labeled hiring data before production use."
    )
    if version == "demo-fallback":
        st.caption("Demo fallback is active until a trained model artifact is saved under models/.")


def render_skill_chips(skills: List[str], kind: str) -> None:
    if not skills:
        st.caption("None found")
        return
    text = "  ".join(f"`{skill}`" for skill in skills[:18])
    if kind == "success":
        st.success(text)
    elif kind == "warning":
        st.warning(text)
    else:
        st.info(text)


def recommendation_label(score: float, threshold: float) -> str:
    if score >= max(0.8, threshold + 0.2):
        return "Strong match - shortlist"
    if score >= threshold:
        return "Potential match - review"
    if score >= max(0.35, threshold - 0.15):
        return "Weak match - consider only if pipeline is thin"
    return "Not recommended"


if __name__ == "__main__":
    main()
