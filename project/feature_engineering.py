"""
Feature engineering for resume screening using TF-IDF vectorization.

This module implements:
1. TF-IDF vectorization with optimized parameters for resumes
2. Similarity computation (cosine similarity)
3. Ranking pipeline
4. Feature importance extraction for explainability

Why TF-IDF for resumes:
- SPARSE DATA: Resumes have variable length, many unique words
- INTERPRETABILITY: TF-IDF weights show which words matter most
- EFFICIENCY: Sparse matrices fit in memory, fast inference
- COMPATIBILITY: Works perfectly with sklearn's linear models
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    TF-IDF based feature engineering for resume screening.
    
    Attributes:
        vectorizer: Fitted sklearn TfidfVectorizer
        feature_names: List of token names (feature vocabulary)
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        strip_accents: str = 'unicode',
        lowercase: bool = True,
        analyzer: str = 'word',
        token_pattern: str = r'(?u)\b\w+\b',
    ):
        """
        Initialize feature engineer with tuned TF-IDF parameters.
        
        Args:
            max_features: Max vocabulary size (5000 captures ~80% of content, saves memory)
            ngram_range: (min_n, max_n) - (1,2) captures individual words + bigrams
                         Bigrams help with skill detection: "machine learning" vs "machine"
            min_df: Ignore terms appearing in < min_df documents (filters noise/typos)
            max_df: Ignore terms appearing in > max_df % of documents (removes common words)
                    0.95 = keep only words that appear in <95% of docs
            sublinear_tf: Apply sublinear TF scaling (dampens term frequency)
                          Prevents long documents from dominating
            
        Parameter tuning rationale:
        - max_features=5000: Balances specificity (avoid OOV) with memory (small deployment)
        - ngram_range=(1,2): Single words + bigrams capture skill phrases well
        - min_df=2: Avoids rare typos that won't help generalization
        - max_df=0.95: Removes ultra-common words (" the ", "a", "experience")
        - sublinear_tf=True: Long resumes shouldn't dominate → fairness
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.analyzer = analyzer
        self.token_pattern = token_pattern
        
        # Initialize vectorizer (unfitted)
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents=strip_accents,
            lowercase=lowercase,
            analyzer=analyzer,
            token_pattern=token_pattern,
        )
        
        self.feature_names = None
        self.is_fitted = False
        
        logger.info("✓ FeatureEngineer initialized")
    
    def fit(self, documents: List[str]):
        """
        Fit TF-IDF vectorizer on training documents.
        
        Args:
            documents: List of text documents (typically combined resume corpus)
            
        Returns:
            self (for method chaining)
        """
        logger.info(f"Fitting TF-IDF vectorizer on {len(documents)} documents...")
        
        # Fit vectorizer
        self.vectorizer.fit(documents)
        
        # Cache feature names for quick lookup
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        
        logger.info(f"✓ Fitted. Vocabulary size: {len(self.feature_names)}")
        logger.info(f"  Features: {self.max_features}")
        logger.info(f"  N-grams: {self.ngram_range}")
        logger.info(f"  Min doc freq: {self.min_df}")
        logger.info(f"  Max doc freq: {self.max_df}")
        
        return self
    
    def transform(self, documents: List[str]):
        """
        Transform documents to TF-IDF sparse matrix.
        
        Args:
            documents: List of text documents
            
        Returns:
            Sparse scipy.sparse matrix of shape (n_docs, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        return self.vectorizer.transform(documents)
    
    def fit_transform(self, documents: List[str]):
        """
        Fit and transform in one step.
        
        Args:
            documents: List of text documents
            
        Returns:
            Sparse TF-IDF matrix
        """
        return self.fit(documents).transform(documents)
    
    def vectorize_single(self, document: str):
        """
        Vectorize a single document (convenience method).
        
        Args:
            document: Single raw text document
            
        Returns:
            Sparse 1D TF-IDF vector
        """
        return self.vectorizer.transform([document])
    
    def get_feature_importance(self, sparse_vector, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract top-K important features (tokens) from a TF-IDF vector.
        
        Args:
            sparse_vector: Single TF-IDF sparse row vector
            top_k: Number of top features to return
            
        Returns:
            List of (feature_name, tfidf_weight) tuples, sorted by weight
        """
        # Convert sparse vector to dense array if needed
        if hasattr(sparse_vector, 'toarray'):
            vector_array = sparse_vector.toarray().flatten()
        else:
            vector_array = sparse_vector
        
        # Get top-K indices
        top_indices = np.argsort(vector_array)[-top_k:][::-1]
        
        # Return feature names with weights
        features = []
        for idx in top_indices:
            if vector_array[idx] > 0:  # Only non-zero features
                features.append((
                    self.feature_names[idx],
                    float(vector_array[idx])
                ))
        
        return features
    
    def save_vectorizer(self, filepath: str):
        """
        Save fitted vectorizer to disk using joblib.
        
        Args:
            filepath: Path to save vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizer")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        logger.info(f"✓ Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath: str):
        """
        Load vectorizer from disk.
        
        Args:
            filepath: Path to saved vectorizer
            
        Returns:
            self (for method chaining)
        """
        self.vectorizer = joblib.load(filepath)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        logger.info(f"✓ Vectorizer loaded from {filepath}")
        return self


class SimilarityRanker:
    """
    Compute similarity between resumes and job descriptions.
    
    Implements cosine similarity for ranking candidates.
    """
    
    def __init__(self, feature_engineer: FeatureEngineer):
        """
        Initialize ranker with equipped feature engineer.
        
        Args:
            feature_engineer: Fitted FeatureEngineer instance
        """
        if not feature_engineer.is_fitted:
            raise ValueError("Feature engineer must be fitted first")
        
        self.feature_engineer = feature_engineer
        logger.info("✓ SimilarityRanker initialized")
    
    def compute_similarity(self, resume_text: str, job_text: str) -> float:
        """
        Compute cosine similarity between resume and job.
        
        Why cosine similarity:
        - Normalized (0-1 range): easy to interpret
        - Efficient on sparse data: O(k) where k = non-zero elements
        - Interpretable: angle between vectors (0° = identical, 90° = orthogonal)
        - No magnitude dependency: short resume vs long resume treated fairly
        
        Args:
            resume_text: Raw resume text
            job_text: Raw job description text
            
        Returns:
            Similarity score 0-1
        """
        # Vectorize
        resume_vec = self.feature_engineer.vectorize_single(resume_text)
        job_vec = self.feature_engineer.vectorize_single(job_text)
        
        # Compute cosine similarity
        similarity = cosine_similarity(resume_vec, job_vec)[0][0]
        
        return float(similarity)
    
    def rank_resumes(
        self,
        resumes: List[str],
        job_text: str,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, str, float]]:
        """
        Rank multiple resumes against a single job.
        
        Args:
            resumes: List of resume texts
            job_text: Job description text
            threshold: Minimum similarity to include (optional)
            
        Returns:
            List of (index, resume_text, similarity_score) tuples,
            sorted by similarity (descending)
        """
        # Vectorize all resumes
        resume_vecs = self.feature_engineer.vectorizer.transform(resumes)
        
        # Vectorize job
        job_vec = self.feature_engineer.vectorize_single(job_text)
        
        # Batch similarity computation (efficient on sparse data)
        similarities = cosine_similarity(resume_vecs, job_vec).flatten()
        
        # Create ranking with metadata
        rankings = [
            (i, resumes[i], similarities[i])
            for i in range(len(resumes))
        ]
        
        # Filter by threshold if specified
        if threshold is not None:
            rankings = [(i, r, s) for i, r, s in rankings if s >= threshold]
        
        # Sort by similarity (descending)
        rankings.sort(key=lambda x: x[2], reverse=True)
        
        return rankings
    
    def get_top_k_matches(
        self,
        resumes: List[str],
        job_text: str,
        k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Get top-K matching resumes for a job.
        
        Args:
            resumes: List of resume texts
            job_text: Job description text
            k: Number of top matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of dicts: {index, score, top_keywords}
        """
        rankings = self.rank_resumes(resumes, job_text, threshold=threshold)
        
        # Get job keywords for reference
        job_vec = self.feature_engineer.vectorize_single(job_text)
        job_keywords = self.feature_engineer.get_feature_importance(job_vec, top_k=5)
        
        # Format results
        results = []
        for rank, (idx, resume, score) in enumerate(rankings[:k]):
            resume_vec = self.feature_engineer.vectorize_single(resume)
            resume_keywords = self.feature_engineer.get_feature_importance(resume_vec, top_k=5)
            
            results.append({
                'rank': rank + 1,
                'resume_index': idx,
                'similarity_score': float(score),
                'top_resume_keywords': resume_keywords,
                'top_job_keywords': job_keywords,
            })
        
        return results


# Testing & Benchmarking
def benchmark_similarity_computation(num_resumes: int = 100):
    """
    Benchmark TF-IDF vectorization and similarity computation.
    
    **Performance expectation**: ~1-5ms for vectorization + similarity of 100 resumes
    """
    import time
    from data_generation import SyntheticDataGenerator
    
    logger.info("Benchmarking TF-IDF + Similarity computation...")
    
    # Generate sample data
    generator = SyntheticDataGenerator()
    resumes = [generator.generate_resume()['raw_text'] for _ in range(num_resumes)]
    job = generator.generate_job_description()['raw_text']
    
    # Fit vectorizer
    feature_engineer = FeatureEngineer(max_features=5000)
    feature_engineer.fit(resumes + [job])
    
    # Benchmark ranking
    ranker = SimilarityRanker(feature_engineer)
    
    start = time.time()
    rankings = ranker.rank_resumes(resumes, job)
    elapsed = time.time() - start
    
    logger.info(f"✓ Benchmark Results:")
    logger.info(f"  Resumes ranked: {len(resumes)}")
    logger.info(f"  Total time: {elapsed*1000:.2f}ms")
    logger.info(f"  Avg per resume: {elapsed*1000/len(resumes):.2f}ms")
    logger.info(f"  Throughput: {len(resumes)/elapsed:.0f} resumes/sec")
    logger.info(f"  Batch of 1K: {len(resumes)/elapsed/1000:.2f}s")
    
    return elapsed


if __name__ == "__main__":
    from data_generation import SyntheticDataGenerator
    
    # Generate sample data
    print("Testing Feature Engineering & Similarity Ranking")
    print("="*60)
    
    generator = SyntheticDataGenerator(seed=42)
    resumes = [generator.generate_resume()['raw_text'] for _ in range(10)]
    job = generator.generate_job_description()['raw_text']
    
    # Test feature engineering
    print("\n1. TF-IDF Vectorization:")
    feature_engineer = FeatureEngineer(max_features=5000)
    feature_engineer.fit(resumes + [job])
    
    vec_job = feature_engineer.vectorize_single(job)
    print(f"   Job vector shape: {vec_job.shape}")
    print(f"   Job non-zero features: {vec_job.nnz}")
    
    job_keywords = feature_engineer.get_feature_importance(vec_job, top_k=10)
    print(f"   Top keywords: {[kw for kw, _ in job_keywords]}")
    
    # Test similarity ranking
    print("\n2. Similarity Ranking:")
    ranker = SimilarityRanker(feature_engineer)
    
    top_matches = ranker.get_top_k_matches(resumes, job, k=5)
    for match in top_matches:
        print(f"   Rank {match['rank']}: Score {match['similarity_score']:.3f}")
    
    # Benchmark
    print("\n3. Performance Benchmark:")
    benchmark_similarity_computation(num_resumes=50)
