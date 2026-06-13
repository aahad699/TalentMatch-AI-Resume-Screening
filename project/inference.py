"""
Model inference and management for production deployment.

This module provides:
1. Inference wrapper for trained model
2. Model versioning and loading
3. Explainability generation
4. Batch inference optimization
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import time

try:
    import joblib
except ImportError:
    joblib = None

try:
    import numpy as np
except ImportError:
    np = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoModelInference:
    """
    Deterministic fallback inference for demos when trained artifacts are absent.

    This keeps the API and Streamlit UI runnable for portfolio use while being
    explicit that it is a transparent heuristic, not a trained classifier.
    """

    def __init__(self):
        try:
            from skill_extraction import SkillExtractor
        except ImportError:
            from .skill_extraction import SkillExtractor

        self.skill_extractor = SkillExtractor(use_tfidf_fallback=False)
        self.metadata = {
            "version": "demo-fallback",
            "model_type": "skill_overlap_tfidf_heuristic",
            "training_date": None,
            "main_metric": {
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "note": "Demo fallback. Train notebooks/02_Model_Training.ipynb for evaluated metrics.",
            },
        }
        logger.warning("No trained model found. Using demo fallback inference.")

    def predict(
        self,
        resume_text: str,
        job_text: str,
        threshold: Optional[float] = None,
    ) -> Dict:
        start = time.time()
        threshold = 0.5 if threshold is None else threshold

        skill_match = self.skill_extractor.match_skills(resume_text, job_text)
        keyword_score = self._keyword_overlap_score(resume_text, job_text)
        score = min(1.0, (0.75 * skill_match["match_score"]) + (0.25 * keyword_score))

        return {
            "match_score": score,
            "match_probability": score,
            "is_match": score >= threshold,
            "inference_time_ms": (time.time() - start) * 1000,
        }

    def predict_with_explanation(
        self,
        resume_text: str,
        job_text: str,
        top_k_keywords: int = 10,
        threshold: Optional[float] = None,
    ) -> Dict:
        prediction = self.predict(resume_text, job_text, threshold)
        skill_match = self.skill_extractor.match_skills(resume_text, job_text)
        top_keywords = self._top_shared_keywords(resume_text, job_text, top_k_keywords)

        matched = ", ".join(skill_match["matched_skills"][:6]) or "general text overlap"
        missing = ", ".join(skill_match["missing_skills"][:5])
        gap_text = f" Missing or weakly represented skills include: {missing}." if missing else ""

        prediction["top_keywords"] = top_keywords
        prediction["explanation"] = (
            "Demo fallback prediction based on extracted skill overlap and shared keywords. "
            f"Matched evidence: {matched}.{gap_text} "
            "Train the notebook model to replace this heuristic with the evaluated ML pipeline."
        )
        return prediction

    def batch_predict(
        self,
        resumes: List[str],
        job_text: str,
        threshold: Optional[float] = None,
    ) -> Dict:
        start = time.time()
        predictions = []
        for index, resume in enumerate(resumes):
            result = self.predict(resume, job_text, threshold)
            predictions.append({
                "resume_index": index,
                "match_probability": result["match_probability"],
                "is_match": result["is_match"],
            })
        predictions.sort(key=lambda item: item["match_probability"], reverse=True)
        elapsed_ms = (time.time() - start) * 1000

        return {
            "predictions": predictions,
            "total_time_ms": elapsed_ms,
            "avg_time_per_resume_ms": elapsed_ms / len(resumes) if resumes else 0.0,
        }

    def get_model_info(self) -> Dict:
        return {
            "model_version": self.metadata["version"],
            "model_type": self.metadata["model_type"],
            "training_date": self.metadata["training_date"],
            "metrics": self.metadata["main_metric"],
            "feature_count": 0,
            "vocabulary_size": len(self.skill_extractor.hard_skills) + len(self.skill_extractor.soft_skills),
        }

    def _keyword_overlap_score(self, resume_text: str, job_text: str) -> float:
        resume_terms = self._important_terms(resume_text)
        job_terms = self._important_terms(job_text)
        if not job_terms:
            return 0.0
        return len(resume_terms & job_terms) / len(job_terms)

    def _top_shared_keywords(self, resume_text: str, job_text: str, top_k: int) -> List[Tuple[str, float]]:
        resume_terms = self._important_terms(resume_text)
        job_terms = self._important_terms(job_text)
        shared = resume_terms & job_terms
        if not shared:
            shared = resume_terms | job_terms

        ranked = sorted(shared, key=lambda term: (resume_text.lower().count(term) + job_text.lower().count(term), term), reverse=True)
        return [(term, 1.0) for term in ranked[:top_k]]

    def _important_terms(self, text: str) -> set:
        stop_words = {
            "and", "are", "for", "from", "has", "have", "the", "this", "that", "with",
            "will", "you", "your", "our", "job", "role", "work", "team", "experience",
            "skills", "required", "preferred", "candidate", "resume", "description",
        }
        return {
            word
            for word in re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.-]{2,}\b", text.lower())
            if word not in stop_words
        }


class ModelInference:
    """
    Lightweight inference wrapper for trained resume-job matching model.
    
    Features:
    - Single model load (no reload per inference)
    - CPU-optimized
    - Fast (<100ms per sample)
    - Explainability support
    """
    
    def __init__(self, model_path: str, metadata_path: str):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to saved joblib pipeline
            metadata_path: Path to metadata.json
        """
        if joblib is None or np is None:
            raise ImportError("joblib and numpy are required to load trained model artifacts")

        # Load model
        self.model = joblib.load(model_path)
        logger.info(f"✓ Model loaded from {model_path}")
        
        # Load metadata
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        logger.info(f"✓ Metadata loaded: v{self.metadata.get('version')}")
        
        # Extract components for explainability
        self.vectorizer = self.model.named_steps['tfidf']
        self.classifier = self.model.named_steps['classifier']
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.coefficients = self.classifier.coef_[0]
        try:
            from skill_extraction import SkillExtractor
        except ImportError:
            from .skill_extraction import SkillExtractor
        self.skill_extractor = SkillExtractor(use_tfidf_fallback=False)
        
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
    
    def predict(
        self,
        resume_text: str,
        job_text: str,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Predict whether resume matches job.
        
        Args:
            resume_text: Raw resume text
            job_text: Raw job description text
            threshold: Confidence threshold for match decision (optional)
            
        Returns:
            {
                'match_score': float (0-1),
                'match_probability': float (model confidence),
                'is_match': bool,
                'inference_time_ms': float,
            }
        """
        start = time.time()
        
        # Combine texts
        combined_text = f"RESUME:\n{resume_text}\n\nJOB:\n{job_text}"
        
        # Vectorize
        vec = self.vectorizer.transform([combined_text])
        
        # Predict
        pred_label = self.classifier.predict(vec)[0]
        pred_proba = self.classifier.predict_proba(vec)[0]
        
        # Handle both binary and multi-class (we use binary, so proba[1] = match probability)
        match_probability = float(pred_proba[1]) if len(pred_proba) > 1 else float(pred_proba[0])
        
        skill_match = self.skill_extractor.match_skills(resume_text, job_text)
        skill_score = float(skill_match.get("match_score", 0.0))
        match_score = (0.45 * match_probability) + (0.55 * skill_score)

        # Determine match with the blended score. The trained probability is
        # still returned, but skill overlap keeps screening recommendations sane.
        is_match = pred_label == 1
        if threshold is not None:
            is_match = match_score >= threshold
        
        elapsed_ms = (time.time() - start) * 1000
        
        return {
            'match_score': match_score,
            'match_probability': match_probability,
            'is_match': is_match,
            'inference_time_ms': elapsed_ms,
        }
    
    def predict_with_explanation(
        self,
        resume_text: str,
        job_text: str,
        top_k_keywords: int = 10,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Predict with explainability metadata.
        
        Returns predictions plus:
        - Top TF-IDF keywords from resume & job
        - Model feature importance
        - Human-readable explanation
        
        Args:
            resume_text: Raw resume text
            job_text: Raw job description text
            top_k_keywords: Number of top keywords to return
            threshold: Confidence threshold
            
        Returns:
            Complete prediction dict with explanation
        """
        # Get base prediction
        prediction = self.predict(resume_text, job_text, threshold)
        
        # Vectorize individually for keyword extraction
        combined_text = f"RESUME:\n{resume_text}\n\nJOB:\n{job_text}"
        vec = self.vectorizer.transform([combined_text])
        
        # Extract top keywords
        top_keywords = self._get_top_keywords(vec, top_k=top_k_keywords)
        
        # Generate explanation
        explanation = self._generate_explanation(prediction, top_keywords)
        
        # Add to prediction
        prediction['top_keywords'] = top_keywords
        prediction['explanation'] = explanation
        
        return prediction
    
    def batch_predict(
        self,
        resumes: List[str],
        job_text: str,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Batch predict for multiple resumes against one job.
        
        Optimized for speed: vectorizes all at once, batch predicts.
        
        Args:
            resumes: List of resume texts
            job_text: Job description text
            threshold: Confidence threshold
            
        Returns:
            List of prediction dicts, sorted by match_probability (descending)
        """
        start = time.time()
        
        # Vectorize all at once
        combined_texts = [f"RESUME:\n{r}\n\nJOB:\n{job_text}" for r in resumes]
        vecs = self.vectorizer.transform(combined_texts)
        
        # Batch predict
        pred_labels = self.classifier.predict(vecs)
        pred_probas = self.classifier.predict_proba(vecs)
        
        # Format results
        results = []
        for i, resume in enumerate(resumes):
            pred_proba = float(pred_probas[i][1]) if pred_probas.shape[1] > 1 else float(pred_probas[i][0])
            is_match = pred_labels[i] == 1
            if threshold is not None:
                is_match = pred_proba >= threshold
            
            results.append({
                'resume_index': i,
                'match_probability': pred_proba,
                'is_match': is_match,
            })
        
        # Sort by match_probability (descending)
        results.sort(key=lambda x: x['match_probability'], reverse=True)
        
        elapsed_ms = (time.time() - start) * 1000
        
        return {
            'predictions': results,
            'total_time_ms': elapsed_ms,
            'avg_time_per_resume_ms': elapsed_ms / len(resumes),
        }
    
    def get_model_info(self) -> Dict:
        """Get model metadata and status."""
        return {
            'model_version': self.metadata.get('version'),
            'model_type': self.metadata.get('model_type'),
            'training_date': self.metadata.get('training_date'),
            'metrics': self.metadata.get('main_metric'),
            'training_samples': self.metadata.get('training_samples'),
            'test_samples': self.metadata.get('test_samples'),
            'notes': self.metadata.get('notes'),
            'feature_count': len(self.feature_names),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
        }
    
    # ========================================================================
    # PRIVATE METHODS FOR EXPLAINABILITY
    # ========================================================================
    
    def _get_top_keywords(self, vec, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract top TF-IDF keywords from a vector."""
        vector_array = vec.toarray().flatten()
        top_indices = np.argsort(vector_array)[-top_k:][::-1]
        
        keywords = []
        for idx in top_indices:
            if vector_array[idx] > 0:
                keywords.append((
                    self.feature_names[idx],
                    float(vector_array[idx])
                ))
        
        return keywords
    
    def _get_top_features_for_prediction(self, top_k: int = 5) -> Tuple[List[str], List[str]]:
        """Get top positive/negative features the model learned."""
        top_pos_idx = np.argsort(self.coefficients)[-top_k:][::-1]
        top_neg_idx = np.argsort(self.coefficients)[:top_k]
        
        positive_features = [self.feature_names[i] for i in top_pos_idx]
        negative_features = [self.feature_names[i] for i in top_neg_idx]
        
        return positive_features, negative_features
    
    def _generate_explanation(self, prediction: Dict, top_keywords: List[Tuple[str, float]]) -> str:
        """Generate human-readable explanation."""
        score = prediction['match_score']
        is_match = prediction['is_match']
        
        if is_match:
            keywords_str = ", ".join([kw for kw, _ in top_keywords[:5]])
            explanation = (
                f"MATCH PREDICTED (confidence: {score:.1%}). "
                f"The resume and job description share relevant keywords: {keywords_str}. "
                "The final score blends the trained model probability with explicit skill overlap "
                "so recommendations remain explainable."
            )
        else:
            explanation = (
                f"NO MATCH PREDICTED (confidence: {1-score:.1%}). "
                f"The resume and job description have limited overlap. "
                f"There may be significant skill or experience gaps."
            )
        
        return explanation


class ModelManager:
    """
    Manage model versions, loading, and registry.
    
    Provides version control and model switching capabilities.
    """
    
    def __init__(self, models_dir: str = "models"):
        """Initialize model manager."""
        self.models_dir = Path(models_dir)
        self.current_model = None
        self.current_version = None
    
    def list_models(self) -> List[str]:
        """List all available model versions."""
        if not self.models_dir.exists():
            return []
        
        versions = []
        for path in self.models_dir.iterdir():
            if path.is_dir() and (path / "pipeline.joblib").exists():
                versions.append(path.name)
        
        return sorted(versions, reverse=True)
    
    def load_model(self, version: str = "latest") -> ModelInference:
        """
        Load a model version.
        
        Args:
            version: Version string or "latest"
            
        Returns:
            ModelInference instance
        """
        if version == "latest":
            available = self.list_models()
            if not available:
                self.current_model = DemoModelInference()
                self.current_version = "demo-fallback"
                return self.current_model
            version = available[0]
        
        model_dir = self.models_dir / version
        if not model_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found")
        
        model_path = model_dir / "pipeline.joblib"
        metadata_path = model_dir / "metadata.json"
        
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Model files missing in {model_dir}")
        
        self.current_model = ModelInference(str(model_path), str(metadata_path))
        self.current_version = version
        
        logger.info(f"✓ Loaded model version {version}")
        
        return self.current_model


if __name__ == "__main__":
    # Test inference (would need actual trained model)
    print("Model Inference Module Ready")
    print("\nUsage:")
    print("  manager = ModelManager()")
    print("  model = manager.load_model('latest')")
    print("  result = model.predict(resume_text, job_text)")
