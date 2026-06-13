"""
Data schemas for Resume Screening System.

Defines Pydantic models for:
- Resume objects
- Job descriptions
- Match labels
- Inference requests/responses
"""

from pydantic import BaseModel, Field, validator
from typing import Any, List, Dict, Optional
from datetime import datetime
from enum import Enum


class ExperienceLevel(str, Enum):
    """Candidate experience level."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"


class Resume(BaseModel):
    """Resume schema."""
    resume_id: str = Field(..., description="Unique resume identifier")
    candidate_name: str = Field(..., description="Candidate full name")
    email: Optional[str] = Field(None, description="Contact email")
    phone: Optional[str] = Field(None, description="Contact phone")
    raw_text: str = Field(..., description="Full resume text")
    experience_years: Optional[float] = Field(None, ge=0, description="Years of experience")
    experience_level: Optional[ExperienceLevel] = Field(None, description="Seniority level")
    summary: Optional[str] = Field(None, description="Professional summary/objective")
    skills_mentioned: Optional[List[str]] = Field(default_factory=list, description="Mentioned skills")
    companies_worked: Optional[List[str]] = Field(default_factory=list, description="Previous employers")
    education: Optional[str] = Field(None, description="Education background")
    
    class Config:
        extra = "allow"  # Allow additional fields
    
    @validator('raw_text')
    def text_not_empty(cls, v):
        if not v or len(v.strip()) < 50:
            raise ValueError("Resume text must contain at least 50 characters")
        return v


class JobDescription(BaseModel):
    """Job description schema."""
    job_id: str = Field(..., description="Unique job ID")
    title: str = Field(..., description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    raw_text: str = Field(..., description="Full job description text")
    required_skills: Optional[List[str]] = Field(default_factory=list, description="Explicitly required skills")
    preferred_skills: Optional[List[str]] = Field(default_factory=list, description="Preferred/nice-to-have skills")
    experience_years_required: Optional[float] = Field(None, ge=0, description="Years of experience required")
    experience_level: Optional[ExperienceLevel] = Field(None, description="Required seniority level")
    location: Optional[str] = Field(None, description="Job location")
    salary_range: Optional[str] = Field(None, description="Salary range if available")
    
    class Config:
        extra = "allow"
    
    @validator('raw_text')
    def text_not_empty(cls, v):
        if not v or len(v.strip()) < 100:
            raise ValueError("Job description must contain at least 100 characters")
        return v


class MatchLabel(BaseModel):
    """Labeled training pair for supervised learning."""
    resume_id: str = Field(..., description="Resume ID")
    job_id: str = Field(..., description="Job ID")
    is_match: bool = Field(..., description="1 = qualified match, 0 = not qualified")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Labeler confidence")
    labeler_notes: Optional[str] = Field(None, description="Notes from human labeler")
    label_timestamp: datetime = Field(default_factory=datetime.utcnow)


class SkillMatch(BaseModel):
    """Extracted skill with confidence."""
    skill_name: str
    source: str = Field(description="Where skill was found: 'resume' or 'jd'")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    context: Optional[str] = Field(None, description="Context sentence containing skill")


class MatchRequest(BaseModel):
    """API request for matching resume to job."""
    resume_text: str = Field(..., description="Resume text (plain text)")
    job_text: str = Field(..., description="Job description text")
    return_top_k_keywords: int = Field(default=5, ge=1, le=20)
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)


class BatchMatchRequest(BaseModel):
    """API request for ranking multiple resumes against one job."""
    resumes: List[str] = Field(..., min_length=1, description="Resume texts to rank")
    job_text: str = Field(..., description="Job description text")
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    return_top_k_keywords: int = Field(default=5, ge=1, le=20)


class MatchResponse(BaseModel):
    """API response for match endpoint."""
    match_score: float = Field(..., ge=0.0, le=1.0, description="Match score 0-1")
    match_probability: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    is_match: bool = Field(..., description="Match/no-match decision")
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    top_keywords_resume: List[tuple] = Field(default_factory=list, description="(keyword, tfidf_weight)")
    top_keywords_job: List[tuple] = Field(default_factory=list, description="(keyword, tfidf_weight)")
    explanation_html: Optional[str] = Field(None, description="HTML-formatted explanation")
    inference_time_ms: float = Field(description="Inference latency in milliseconds")


class BatchMatchItem(BaseModel):
    """One ranked candidate in a batch match response."""
    rank: int
    resume_index: int
    match_score: float = Field(ge=0.0, le=1.0)
    is_match: bool
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    recommendation: str
    inference_time_ms: float


class BatchMatchResponse(BaseModel):
    """API response for batch candidate ranking."""
    rankings: List[BatchMatchItem]
    total_resumes: int
    average_score: float


class SkillExtractionRequest(BaseModel):
    """Request for skill extraction endpoint."""
    text: str = Field(..., description="Text to extract skills from")


class SkillExtractionResponse(BaseModel):
    """Response from skill extraction endpoint."""
    technical_skills: List[SkillMatch]
    soft_skills: List[SkillMatch]
    extracted_keywords: List[tuple] = Field(default_factory=list, description="(keyword, tfidf_weight)")
    processing_time_ms: float


class ModelMetadata(BaseModel):
    """Model versioning and metadata."""
    model_id: str
    model_type: str = Field(description="e.g., 'logistic_regression', 'svm', 'random_forest'")
    version: str = Field(description="Semantic version (e.g., '1.0.0')")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    training_date: Optional[datetime] = None
    main_metric: Optional[float] = Field(None, description="Primary evaluation metric (F1/AUC)")
    training_samples: int = Field(description="Number of training samples")
    feature_count: int = Field(description="Number of TF-IDF features")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = Field(None)


if __name__ == "__main__":
    # Example usage
    resume = Resume(
        resume_id="R001",
        candidate_name="John Doe",
        email="john@example.com",
        raw_text="Senior Python Developer with 5 years experience in ML and backend services.",
        experience_years=5.0,
        experience_level=ExperienceLevel.SENIOR,
        skills_mentioned=["Python", "Machine Learning", "FastAPI", "Docker"]
    )
    print("Resume example:", resume.json(indent=2))
    
    job = JobDescription(
        job_id="J001",
        title="Machine Learning Engineer",
        company="TechCorp",
        raw_text="We are seeking an experienced ML engineer with strong Python skills.",
        required_skills=["Python", "Machine Learning", "Statistics"],
        experience_level=ExperienceLevel.MID
    )
    print("Job example:", job.json(indent=2))
