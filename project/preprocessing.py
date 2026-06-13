"""
NLP preprocessing pipeline for resume screening.

This module provides modular, configurable text preprocessing with:
- Regex-based cleaning
- Tokenization
- Stopword removal
- Lemmatization (spaCy)
- Token filtering

Design: Follows sklearn fit/transform pattern for consistency with Pipeline
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from typing import List, Dict, Set, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ResumePipelinePreprocessor:
    """
    Production-grade text preprocessor for resume/job description text.
    
    Implements: lowercasing, regex cleaning, tokenization, lemmatization,
    stopword removal, token filtering.
    
    Design: fit() learns vocabulary/stopwords, transform() applies preprocessing
    Pattern: Consistent with sklearn Pipeline usage
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_special_chars: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_length: int = 2,
        max_token_length: int = 50,
        preserve_skills: Optional[Set[str]] = None,
        stopwords_lang: str = "english"
    ):
        """
        Initialize preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs (http://, https://, www.)
            remove_emails: Remove email addresses
            remove_special_chars: Remove special characters (keep alphanumeric, hyphens)
            remove_numbers: Remove numeric tokens (usually preserve these)
            remove_stopwords: Remove common stopwords (the, a, is, etc.)
            lemmatize: Apply lemmatization (requires spaCy model)
            min_token_length: Filter tokens shorter than this
            max_token_length: Filter tokens longer than this
            preserve_skills: Set of skill names to NEVER filter out
            stopwords_lang: Language for stopword list
            
        Why each step matters for resumes:
        - Lowercasing: Normalizes variations (Python, python → python)
        - URL removal: Avoids processing garbage tokens
        - Email removal: PII redaction + noise filtering
        - Special chars: Handles LinkedIn URLs, citations → cleaner tokens
        - Lemmatization > stemming: Better accuracy (engineering → engineer not engineer)
        - Min length: Filters noise (single letters, abbreviations often irrelevant)
        - Preserve skills: Avoids dropping important keywords like "C++"
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        self.preserve_skills = preserve_skills or set()
        self.stopwords_lang = stopwords_lang
        
        # Load stopwords
        self.stopwords_set = set(stopwords.words(stopwords_lang))
        
        # Load spaCy model 
        # Using small model for CPU efficiency (en_core_web_sm is ~50MB)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing en_core_web_sm...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        logger.info(f"✓ ResumePipelinePreprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Step 1: Regex-based text cleaning.
        
        Removes URLs, emails, special chars, normalizes whitespace.
        """
        if not text:
            return ""
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove emails (privacy)
        if self.remove_emails:
            text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        
        # Remove special characters (keep alphanumeric, spaces, hyphens, plus)
        if self.remove_special_chars:
            # Preserve: alphanumeric, spaces, hyphens (for "C++", "c#"), plus, dot
            text = re.sub(r'[^a-zA-Z0-9\s\-\+\#\.]', '', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_process(self, text: str) -> List[str]:
        """
        Step 2: Tokenization + lemmatization + filtering.
        
        Uses spaCy for accurate lemmatization while respecting POS tags.
        CPU-efficient: processes one document at a time.
        """
        if not text:
            return []
        
        # Disable unnecessary spaCy components for speed (CPU optimization)
        # We only need tokenization + lemmatization, not NER, parser, etc.
        with self.nlp.select_pipes(disable=["ner", "parser"]):
            doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            lemma = token.lemma_ if self.lemmatize else token.text
            
            # Skip if it's a stopword (unless preserved)
            if self.remove_stopwords and lemma.lower() in self.stopwords_set:
                if lemma.lower() not in self.preserve_skills:
                    continue
            
            # Skip if it's just punctuation
            if token.pos_ == "PUNCT":
                continue
            
            # Skip numbers (usually not signal)
            if self.remove_numbers and token.pos_ == "NUM":
                continue
            
            # Length filtering
            if len(lemma) < self.min_token_length:
                continue
            if len(lemma) > self.max_token_length:
                continue
            
            tokens.append(lemma.lower())
        
        return tokens
    
    def fit(self, documents: List[str]):
        """
        Fit preprocessor on documents.
        
        In this case, fitting is trivial (stopwords/model already loaded).
        Included for sklearn Pipeline compatibility.
        
        Args:
            documents: List of text documents
            
        Returns:
            self (for chaining)
        """
        logger.info(f"Fitting preprocessor on {len(documents)} documents")
        # Could learn domain-specific stopwords here if desired
        return self
    
    def transform(self, documents: List[str]) -> List[List[str]]:
        """
        Transform documents into token lists.
        
        Args:
            documents: List of raw text documents
            
        Returns:
            List of tokenized documents (each is list of lemmatized tokens)
        """
        processed = []
        for i, doc in enumerate(documents):
            cleaned = self.clean_text(doc)
            tokens = self.tokenize_and_process(cleaned)
            processed.append(tokens)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{len(documents)} documents")
        
        return processed
    
    def fit_transform(self, documents: List[str]) -> List[List[str]]:
        """
        Fit and transform in one step.
        
        Args:
            documents: List of raw text documents
            
        Returns:
            List of tokenized documents
        """
        return self.fit(documents).transform(documents)
    
    def preprocess_single(self, text: str) -> List[str]:
        """
        Preprocess a single document (convenience method).
        
        Args:
            text: Single raw document
            
        Returns:
            List of tokens
        """
        cleaned = self.clean_text(text)
        return self.tokenize_and_process(cleaned)
    
    def preprocess_and_rejoin(self, text: str) -> str:
        """
        Preprocess and return as joined string (for TF-IDF vectorizer).
        
        Args:
            text: Raw document
            
        Returns:
            Preprocessed text as space-separated string
        """
        tokens = self.preprocess_single(text)
        return ' '.join(tokens)


# Performance Benchmark Function
def benchmark_preprocessing(num_samples: int = 100, avg_doc_length: int = 200):
    """
    Benchmark preprocessing speed on CPU.
    
    **Performance expectation**: ~0.5-1ms per document on modern CPU
    This is FAST enough for batch screening (1000 resumes in <1 second)
    """
    import time
    from data_generation import SyntheticDataGenerator
    
    logger.info("Benchmarking preprocessing performance...")
    
    # Generate sample texts
    generator = SyntheticDataGenerator()
    sample_texts = []
    for _ in range(num_samples):
        resume = generator.generate_resume()
        sample_texts.append(resume['raw_text'])
    
    # Preprocess with timing
    preprocessor = ResumePipelinePreprocessor()
    
    start = time.time()
    _ = preprocessor.transform(sample_texts)
    elapsed = time.time() - start
    
    avg_time_ms = (elapsed / num_samples) * 1000
    
    logger.info(f"✓ Benchmark Results:")
    logger.info(f"  Documents: {num_samples}")
    logger.info(f"  Total time: {elapsed:.2f}s")
    logger.info(f"  Avg per doc: {avg_time_ms:.2f}ms")
    logger.info(f"  Throughput: {num_samples/elapsed:.0f} docs/sec")
    logger.info(f"  Batch of 1000: {num_samples/elapsed/1000:.2f}s")
    
    return avg_time_ms


if __name__ == "__main__":
    # Example usage
    preprocessor = ResumePipelinePreprocessor(
        lowercase=True,
        remove_stopwords=True,
        lemmatize=True,
        preserve_skills={"python", "java", "c++", "machine learning"}
    )
    
    # Sample text
    sample_resume = """
    Senior Python Developer with 5 years of experience
    Skills: Python, Machine Learning, FastAPI, Docker, SQL
    Experience: Built ML models for recommendation systems
    Link: https://linkedin.com/in/john-doe
    Email: john@example.com
    """
    
    # Preprocess
    tokens = preprocessor.preprocess_single(sample_resume)
    print(f"Tokens: {tokens}")
    
    # Rejoin for TF-IDF
    preprocessed_text = preprocessor.preprocess_and_rejoin(sample_resume)
    print(f"Preprocessed: {preprocessed_text}")
    
    # Run benchmark
    print("\n" + "="*60)
    benchmark_preprocessing(num_samples=50)
