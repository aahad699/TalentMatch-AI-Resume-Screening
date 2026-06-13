"""
FastAPI service for Resume Screening System.

Provides REST API endpoints:
- POST /match: Match resume to job
- POST /extract-skills: Extract skills from text
- GET /health: Service health check

CPU-optimized for production deployment.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import sys
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

from inference import ModelManager
from skill_extraction import SkillExtractor
from schemas import (
    MatchRequest, MatchResponse, SkillExtractionRequest, SkillExtractionResponse,
    SkillMatch, BatchMatchRequest, BatchMatchResponse, BatchMatchItem
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Resume Screening API",
    description="AI-powered resume-job matching system",
    version="1.0.0",
)

# CORS configuration (allow requests from frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global resources (loaded once at startup)
model_manager = None
skill_extractor = None
start_time = None


@app.on_event("startup")
async def startup_event():
    """Initialize ML models on app startup (not per request)."""
    global model_manager, skill_extractor, start_time
    
    start_time = time.time()
    logger.info("Starting Resume Screening API...")
    
    try:
        # Load model
        model_manager = ModelManager(models_dir="./models")
        model = model_manager.load_model("latest")
        logger.info("✓ Model loaded successfully")
        
        # Initialize skill extractor
        skill_extractor = SkillExtractor()
        logger.info("✓ Skill extractor initialized")
        
        startup_time = time.time() - start_time
        logger.info(f"✓ API ready in {startup_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Resume Screening API")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.
    
    Returns: Service status, model version, uptime
    """
    try:
        model_info = model_manager.current_model.get_model_info()
        uptime_seconds = time.time() - start_time
        
        return {
            "status": "healthy",
            "uptime_seconds": uptime_seconds,
            "model_version": model_info['model_version'],
            "model_type": model_info['model_type'],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.post("/match", response_model=MatchResponse)
async def match_resume_to_job(request: MatchRequest) -> MatchResponse:
    """
    Match a resume to a job description.
    
    Computes:
    - Match score (0-1)
    - Match decision (boolean)
    - Top matching keywords
    - Explanation for HR teams
    
    Args:
        request: MatchRequest with resume_text and job_text
        
    Returns:
        MatchResponse with detailed matching results
    """
    try:
        # Validate inputs
        if not request.resume_text or len(request.resume_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Resume text must be at least 50 characters"
            )
        
        if not request.job_text or len(request.job_text.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Job description must be at least 100 characters"
            )
        
        # Get prediction with explanation
        inference_start = time.time()
        prediction = model_manager.current_model.predict_with_explanation(
            request.resume_text,
            request.job_text,
            top_k_keywords=request.return_top_k_keywords,
            threshold=request.threshold,
        )
        inference_time = (time.time() - inference_start) * 1000
        
        # Extract skills
        resume_skills = skill_extractor.extract_from_text(request.resume_text)
        job_skills = skill_extractor.extract_from_text(request.job_text)
        
        # Calculate skill match
        skill_match_result = skill_extractor.match_skills(
            request.resume_text,
            request.job_text
        )
        
        # Convert skill matches to response format
        matched_skills = skill_match_result['matched_skills']
        missing_skills = skill_match_result['missing_skills']
        
        # Format keywords
        top_keywords = [
            (keyword, float(weight))
            for keyword, weight in prediction.get('top_keywords', [])
        ]
        
        # Build response
        response = MatchResponse(
            match_score=prediction['match_score'],
            match_probability=prediction['match_probability'],
            is_match=prediction['is_match'],
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            top_keywords_resume=top_keywords,
            top_keywords_job=[],  # Could extract separately if needed
            explanation_html=_format_explanation_html(prediction),
            inference_time_ms=prediction['inference_time_ms'],
        )
        
        logger.info(f"Match prediction: score={response.match_score:.3f}, is_match={response.is_match}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Match endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-match", response_model=BatchMatchResponse)
async def batch_match_resumes(request: BatchMatchRequest) -> BatchMatchResponse:
    """
    Rank multiple resumes against one job description.

    This endpoint powers the recruiter-style screening workflow: one job in,
    many resumes ranked by score with matched and missing skills.
    """
    try:
        if not request.job_text or len(request.job_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Job description must be at least 50 characters")

        rows = []
        for index, resume_text in enumerate(request.resumes):
            if not resume_text or len(resume_text.strip()) < 20:
                continue

            prediction = model_manager.current_model.predict_with_explanation(
                resume_text,
                request.job_text,
                top_k_keywords=request.return_top_k_keywords,
                threshold=request.threshold,
            )
            skill_match = skill_extractor.match_skills(resume_text, request.job_text)
            rows.append({
                "resume_index": index,
                "match_score": float(prediction["match_score"]),
                "is_match": bool(prediction["is_match"]),
                "matched_skills": skill_match["matched_skills"],
                "missing_skills": skill_match["missing_skills"],
                "recommendation": _recommendation_label(float(prediction["match_score"]), request.threshold or 0.5),
                "inference_time_ms": float(prediction["inference_time_ms"]),
            })

        rows.sort(key=lambda item: item["match_score"], reverse=True)
        rankings = [
            BatchMatchItem(rank=rank, **row)
            for rank, row in enumerate(rows, start=1)
        ]
        average_score = sum(row.match_score for row in rankings) / len(rankings) if rankings else 0.0

        return BatchMatchResponse(
            rankings=rankings,
            total_resumes=len(rankings),
            average_score=average_score,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch match endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-skills", response_model=SkillExtractionResponse)
async def extract_skills(request: SkillExtractionRequest) -> SkillExtractionResponse:
    """
    Extract skills from text.
    
    Identifies technical and soft skills using hybrid extraction:
    1. Dictionary matching (precise)
    2. Fuzzy matching (typo-tolerance)
    3. TF-IDF fallback (new skills)
    
    Args:
        request: Text to extract skills from
        
    Returns:
        SkillExtractionResponse with hard and soft skills
    """
    try:
        if not request.text or len(request.text.strip()) < 20:
            raise HTTPException(
                status_code=400,
                detail="Text must be at least 20 characters"
            )
        
        processing_start = time.time()
        
        # Extract skills
        extraction_result = skill_extractor.extract_from_text(request.text)
        
        # Convert to response format
        hard_skills = [
            SkillMatch(
                skill_name=skill['name'],
                source="extraction",
                confidence=skill['confidence'],
                context=None,
            )
            for skill in extraction_result['hard_skills']
        ]
        
        soft_skills = [
            SkillMatch(
                skill_name=skill['name'],
                source="extraction",
                confidence=skill['confidence'],
                context=None,
            )
            for skill in extraction_result['soft_skills']
        ]
        
        processing_time = (time.time() - processing_start) * 1000
        
        response = SkillExtractionResponse(
            technical_skills=hard_skills,
            soft_skills=soft_skills,
            extracted_keywords=[],  # Could add TF-IDF keywords here
            processing_time_ms=processing_time,
        )
        
        logger.info(f"Extracted {len(hard_skills)} hard skills, {len(soft_skills)} soft skills")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Skill extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API documentation link."""
    return {
        "message": "Resume Screening API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
        "version": "1.0.0",
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _format_explanation_html(prediction: Dict) -> str:
    """Format explanation as HTML for display in UI."""
    explanation = prediction.get('explanation', 'No explanation available')
    
    html = f"""
    <div style="font-family: Arial, sans-serif; color: #333;">
        <p>{explanation}</p>
    </div>
    """
    
    return html


def _recommendation_label(score: float, threshold: float = 0.5) -> str:
    """Turn a numeric score into a recruiter-friendly label."""
    if score >= max(0.8, threshold + 0.2):
        return "Strong match - shortlist"
    if score >= threshold:
        return "Potential match - review"
    if score >= max(0.35, threshold - 0.15):
        return "Weak match - consider only if pipeline is thin"
    return "Not recommended"


# Custom exception handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run locally for testing
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
