"""
Lightweight resume parsing utilities.

This is not a full ATS parser. It extracts practical profile signals that make
the demo feel more recruiter-friendly: contact info, likely name, sections,
years of experience, and skill evidence.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


SECTION_NAMES = [
    "summary",
    "professional summary",
    "experience",
    "work experience",
    "skills",
    "technical skills",
    "education",
    "projects",
    "certifications",
]


def parse_resume_profile(text: str, skills: Optional[List[str]] = None) -> Dict:
    """Extract a compact candidate profile from raw resume text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "candidate_name": _guess_name(lines),
        "email": _find_email(text),
        "phone": _find_phone(text),
        "years_experience": _find_years_experience(text),
        "sections": _find_sections(lines),
        "top_skills": (skills or [])[:10],
    }


def _guess_name(lines: List[str]) -> str:
    for line in lines[:8]:
        cleaned = re.sub(r"[^A-Za-z\s.-]", "", line).strip()
        words = cleaned.split()
        if 2 <= len(words) <= 4 and not any(word.lower() in SECTION_NAMES for word in words):
            if all(word[:1].isupper() for word in words if word):
                return cleaned
    return "Candidate"


def _find_email(text: str) -> Optional[str]:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None


def _find_phone(text: str) -> Optional[str]:
    match = re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", text)
    return match.group(0).strip() if match else None


def _find_years_experience(text: str) -> Optional[int]:
    matches = re.findall(r"(\d{1,2})\+?\s*(?:years|yrs)\s+(?:of\s+)?experience", text, flags=re.I)
    if not matches:
        matches = re.findall(r"experience\s*(?:of\s*)?(\d{1,2})\+?\s*(?:years|yrs)", text, flags=re.I)
    years = [int(value) for value in matches if int(value) < 50]
    return max(years) if years else None


def _find_sections(lines: List[str]) -> List[str]:
    found = []
    section_set = {name.lower() for name in SECTION_NAMES}
    for line in lines:
        normalized = re.sub(r"[^A-Za-z\s]", "", line).strip().lower()
        if normalized in section_set:
            found.append(normalized.title())
    return sorted(set(found))
