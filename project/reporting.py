"""
Report helpers for resume matching results.

The functions here avoid heavyweight dependencies so exports work in the
portfolio demo environment as well as the full ML environment.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, Dict, Iterable, List


def build_single_match_report(
    resume_name: str,
    job_name: str,
    prediction: Dict[str, Any],
    skill_match: Dict[str, Any],
    recommendation: str,
) -> Dict[str, Any]:
    """Create a structured report dictionary for one resume-job match."""
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "resume": resume_name,
        "job": job_name,
        "recommendation": recommendation,
        "match_score": round(float(prediction.get("match_score", 0.0)), 4),
        "match_probability": round(float(prediction.get("match_probability", 0.0)), 4),
        "is_match": bool(prediction.get("is_match", False)),
        "matched_skills": skill_match.get("matched_skills", []),
        "missing_skills": skill_match.get("missing_skills", []),
        "top_keywords": prediction.get("top_keywords", []),
        "explanation": prediction.get("explanation", ""),
        "inference_time_ms": round(float(prediction.get("inference_time_ms", 0.0)), 2),
    }


def report_to_markdown(report: Dict[str, Any]) -> str:
    """Render a structured report as readable markdown."""
    matched = ", ".join(report.get("matched_skills", [])) or "None"
    missing = ", ".join(report.get("missing_skills", [])) or "None"
    keywords = ", ".join(str(item[0]) for item in report.get("top_keywords", [])) or "None"

    return f"""# Candidate Match Report

Generated: {report.get("generated_at")}

Resume: {report.get("resume")}
Job: {report.get("job")}

Recommendation: {report.get("recommendation")}
Match Score: {report.get("match_score", 0):.1%}
Decision: {"Match" if report.get("is_match") else "Needs Review"}

Matched Skills:
{matched}

Missing Skills:
{missing}

Top Keywords:
{keywords}

Explanation:
{report.get("explanation", "")}
"""


def report_to_json_bytes(report: Dict[str, Any]) -> bytes:
    """Serialize a report to pretty JSON bytes."""
    return json.dumps(report, indent=2).encode("utf-8")


def rankings_to_csv_bytes(rankings: Iterable[Dict[str, Any]]) -> bytes:
    """Serialize batch ranking rows to CSV bytes."""
    output = io.StringIO()
    fieldnames = [
        "rank",
        "candidate",
        "match_score",
        "semantic_score",
        "recommendation",
        "must_have_coverage",
        "summary",
        "must_have_matched",
        "must_have_missing",
        "interview_focus",
        "matched_skills",
        "missing_skills",
        "inference_time_ms",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rankings:
        scorecard = row.get("scorecard", {})
        writer.writerow({
            "rank": row.get("rank"),
            "candidate": row.get("candidate"),
            "match_score": row.get("match_score"),
            "semantic_score": row.get("semantic_score"),
            "recommendation": row.get("recommendation"),
            "must_have_coverage": scorecard.get("must_have_coverage"),
            "summary": row.get("summary"),
            "must_have_matched": ", ".join(scorecard.get("must_have_matched", [])),
            "must_have_missing": ", ".join(scorecard.get("must_have_missing", [])),
            "interview_focus": " | ".join(scorecard.get("interview_questions", [])),
            "matched_skills": ", ".join(row.get("matched_skills", [])),
            "missing_skills": ", ".join(row.get("missing_skills", [])),
            "inference_time_ms": row.get("inference_time_ms"),
        })
    return output.getvalue().encode("utf-8")


def report_to_pdf_bytes(report: Dict[str, Any]) -> bytes:
    """
    Create a small valid PDF with report text.

    This intentionally uses a minimal PDF writer rather than a dependency. It is
    plain but reliable for portfolio export.
    """
    text = report_to_markdown(report)
    lines = _wrap_lines(text.replace("# ", ""), width=88)[:45]
    content_lines = ["BT", "/F1 10 Tf", "50 780 Td", "14 TL"]
    for index, line in enumerate(lines):
        escaped = _escape_pdf_text(line)
        if index == 0:
            content_lines.append(f"({escaped}) Tj")
        else:
            content_lines.append(f"T* ({escaped}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects: List[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def _wrap_lines(text: str, width: int) -> List[str]:
    lines: List[str] = []
    for raw_line in text.splitlines():
        words = raw_line.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            if len(current) + len(word) + 1 > width:
                lines.append(current)
                current = word
            else:
                current += " " + word
        lines.append(current)
    return lines


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
