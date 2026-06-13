"""
Recruiter-style candidate summaries generated from deterministic signals.

No external LLM is required. This keeps the project easy to run and honest for
portfolio demos while still producing polished decision support text.
"""

from __future__ import annotations

from typing import Dict, List


def build_candidate_summary(
    profile: Dict,
    score: float,
    recommendation: str,
    matched_skills: List[str],
    missing_skills: List[str],
) -> str:
    """Create a concise recruiter-facing summary."""
    name = profile.get("candidate_name") or "The candidate"
    years = profile.get("years_experience")
    years_text = f" with about {years} years of experience" if years else ""
    matched_text = _join_skills(matched_skills[:5], "relevant role skills")
    missing_text = _join_skills(missing_skills[:4], "no major required skills detected as missing")

    if score >= 0.8:
        fit = "appears to be a strong fit"
    elif score >= 0.5:
        fit = "appears to be a potential fit"
    else:
        fit = "needs careful review before moving forward"

    return (
        f"{name}{years_text} {fit} for this role. "
        f"The strongest evidence is overlap in {matched_text}. "
        f"Main gaps: {missing_text}. "
        f"Recommendation: {recommendation}."
    )


def _join_skills(skills: List[str], fallback: str) -> str:
    if not skills:
        return fallback
    if len(skills) == 1:
        return skills[0]
    return ", ".join(skills[:-1]) + f", and {skills[-1]}"
