"""
Optional semantic similarity for resume-job matching.

If sentence-transformers is installed, this module uses embeddings. Otherwise
it falls back to a lightweight sklearn HashingVectorizer similarity so the app
continues to run without large model downloads.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def _load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(resume_text: str, job_text: str) -> Dict:
    """Return a similarity score and the method used."""
    model = _load_embedding_model()
    if model is not None:
        embeddings = model.encode([resume_text, job_text], normalize_embeddings=True)
        score = float(embeddings[0] @ embeddings[1])
        return {"score": max(0.0, min(1.0, score)), "method": "sentence-transformers/all-MiniLM-L6-v2"}

    try:
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return {"score": 0.0, "method": "unavailable"}

    vectorizer = HashingVectorizer(n_features=2**14, alternate_sign=False, ngram_range=(1, 2), norm="l2")
    vectors = vectorizer.transform([resume_text, job_text])
    score = float(cosine_similarity(vectors[0], vectors[1])[0][0])
    return {"score": max(0.0, min(1.0, score)), "method": "hashed lexical similarity"}
