"""
Hybrid skill extraction from resume and job description text.

This module implements THREE-LAYER skill extraction:
1. Rule-based matching against curated skill dictionaries (FAST, ACCURATE)
2. TF-IDF keyword extraction (FLEXIBLE, catches new skills)
3. Fuzzy matching for typos (ERROR-TOLERANT)

Design rationale:
- Dictionary matching is precise but incomplete (doesn't catch all variations/new skills)
- TF-IDF is flexible but noisy (picks up non-skills if document-heavy)
- Hybrid approach combines benefits: precision + recall + robustness

CPU-optimized: In-memory dictionaries, no external API calls
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from fuzzywuzzy import fuzz
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SKILL DICTIONARIES
# ============================================================================
# In production, these would be loaded from curated sources like:
# - JANZZ skill ontology (https://www.janzz.jobs/)
# - O*NET database
# - Custom domain-specific lists

TECHNICAL_SKILLS = {
    # Programming Languages
    "Python", "Java", "C++", "JavaScript", "Go", "Rust", "Ruby", "PHP", "Kotlin", "R",
    "C#", ".NET", "Scala", "Swift", "Objective-C", "TypeScript", "SQL", "VB.NET",
    
    # Data & ML
    "Machine Learning", "Data Science", "TensorFlow", "PyTorch", "scikit-learn",
    "Pandas", "NumPy", "Spark", "Hadoop", "Kafka", "Deep Learning",
    
    # Databases
    "SQL", "MongoDB", "PostgreSQL", "MySQL", "Redis", "Cassandra", "DynamoDB",
    "Elasticsearch", "Firebase", "NoSQL", "Graph Databases",
    
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes", "Docker",
    "CI/CD", "Jenkins", "GitHub Actions", "Terraform", "Infrastructure as Code",
    
    # Frameworks & Libraries
    "FastAPI", "Flask", "Django", "Spring Boot", "React", "Vue.js", "Angular",
    "Node.js", "Express", "Streamlit", "Plotly", "Scikit-learn",
    
    # Big Data & Tools
    "Airflow", "Tableau", "Power BI", "Looker", "Datadog", "Grafana",
    "Jupyter", "Git", "Linux", "Unix", "Shell", "Bash",
}

SOFT_SKILLS = {
    "Leadership", "Communication", "Teamwork", "Project Management",
    "Problem Solving", "Critical Thinking", "Adaptability", "Attention to Detail",
    "Mentoring", "Negotiation", "Presentation", "Time Management", "Collaboration",
}

# Skill aliases (maps common variations to canonical form)
SKILL_ALIASES = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "ds": "data science",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "dl": "deep learning",
    "c++": "c++",
    "c#": "c#",
    "js": "javascript",
    "ts": "typescript",
    "aws": "aws",
    "gcp": "google cloud",
    "k8s": "kubernetes",
    "devops": "devops",
}


class SkillExtractor:
    """
    Multi-layer skill extraction from text.
    
    Public methods:
    - extract_from_text(): Get all skills (rule + TF-IDF)
    - extract_hard_skills(): Technical/hard skills only
    - extract_soft_skills(): Soft/interpersonal skills only
    - match_skills(): Match resume skills to job required skills
    - identify_missing_skills(): Skills required but not found in resume
    """
    
    def __init__(
        self,
        fuzzy_threshold: int = 80,
        min_keyword_frequency: int = 1,
        use_tfidf_fallback: bool = True,
    ):
        """
        Initialize skill extractor.
        
        Args:
            fuzzy_threshold: Confidence threshold for fuzzy matching (0-100)
                            80 = 80% match required (good balance)
            min_keyword_frequency: Minimum TF-IDF frequency for keyword inclusion
            use_tfidf_fallback: Include TF-IDF keywords as fallback
        """
        self.hard_skills = TECHNICAL_SKILLS
        self.soft_skills = SOFT_SKILLS
        self.skill_aliases = SKILL_ALIASES
        self.fuzzy_threshold = fuzzy_threshold
        self.min_keyword_frequency = min_keyword_frequency
        self.use_tfidf_fallback = use_tfidf_fallback
        
        # Create lowercase versions for fast lookup
        self.hard_skills_lower = {s.lower(): s for s in self.hard_skills}
        self.soft_skills_lower = {s.lower(): s for s in self.soft_skills}
        
        logger.info("✓ SkillExtractor initialized")
    
    def extract_from_text(
        self,
        text: str,
        skill_type: str = "all"
    ) -> Dict[str, List[Dict]]:
        """
        Extract skills from text using hybrid approach.
        
        Args:
            text: Raw document text
            skill_type: "all", "hard", or "soft"
            
        Returns:
            Dict with 'hard_skills' and 'soft_skills' lists.
            Each skill dict contains: {name, confidence, method}
            - method: how it was found ("dictionary", "fuzzy", or "tfidf")
        """
        text_lower = text.lower()
        
        # Step 1: Exact dictionary matching
        hard_matches = self._extract_by_dictionary(text_lower, self.hard_skills_lower)
        soft_matches = self._extract_by_dictionary(text_lower, self.soft_skills_lower)
        
        # Step 2: Fuzzy matching (catch typos/variations)
        hard_fuzzy = self._extract_by_fuzzy_matching(text_lower, self.hard_skills)
        soft_fuzzy = self._extract_by_fuzzy_matching(text_lower, self.soft_skills)
        
        # Step 3: TF-IDF fallback (catch new skills)
        if self.use_tfidf_fallback:
            hard_tfidf = self._extract_by_tfidf(text_lower, self.hard_skills)
            soft_tfidf = self._extract_by_tfidf(text_lower, self.soft_skills)
        else:
            hard_tfidf = []
            soft_tfidf = []
        
        # Merge results (avoid duplicates)
        hard_skills = self._merge_skill_results(hard_matches + hard_fuzzy + hard_tfidf)
        soft_skills = self._merge_skill_results(soft_matches + soft_fuzzy + soft_tfidf)
        
        # Filter by skill_type
        if skill_type == "hard":
            return {"hard_skills": hard_skills, "soft_skills": []}
        elif skill_type == "soft":
            return {"hard_skills": [], "soft_skills": soft_skills}
        else:
            return {"hard_skills": hard_skills, "soft_skills": soft_skills}
    
    def extract_hard_skills(self, text: str) -> List[Dict]:
        """Extract only technical/hard skills."""
        return self.extract_from_text(text, skill_type="hard")["hard_skills"]
    
    def extract_soft_skills(self, text: str) -> List[Dict]:
        """Extract only soft/interpersonal skills."""
        return self.extract_from_text(text, skill_type="soft")["soft_skills"]
    
    def match_skills(self, resume_text: str, job_text: str) -> Dict:
        """
        Match resume skills to job requirements.
        
        Returns:
            {
                'matched_skills': [skills found in both],
                'missing_skills': [job skills not in resume],
                'extra_skills': [resume skills not needed],
                'match_score': 0-1 confidence
            }
        """
        resume_skills = self.extract_from_text(resume_text)
        job_skills = self.extract_from_text(job_text)
        
        # Get skill names only (lowercase for comparison)
        resume_hard = {s['name'].lower() for s in resume_skills['hard_skills']}
        resume_soft = {s['name'].lower() for s in resume_skills['soft_skills']}
        resume_all = resume_hard | resume_soft
        
        job_hard = {s['name'].lower() for s in job_skills['hard_skills']}
        job_soft = {s['name'].lower() for s in job_skills['soft_skills']}
        job_all = job_hard | job_soft
        
        # Calculate matches
        matched = resume_all & job_all
        missing = job_all - resume_all
        extra = resume_all - job_all
        
        # Match score: % of job skills found in resume
        match_score = len(matched) / len(job_all) if job_all else 0.0
        
        # Boost score if hard skills match (more important)
        hard_match_score = len(resume_hard & job_hard) / len(job_hard) if job_hard else 0.0
        
        # Weighted average: hard skills worth more
        final_score = 0.7 * hard_match_score + 0.3 * match_score
        
        return {
            'matched_skills': sorted(list(matched)),
            'missing_skills': sorted(list(missing)),
            'extra_skills': sorted(list(extra)),
            'match_score': min(1.0, final_score),
            'hard_match_score': hard_match_score,
        }
    
    def identify_missing_skills(self, resume_text: str, job_text: str) -> List[str]:
        """
        Identify skills required by job but missing in resume.
        
        Returns:
            Sorted list of missing skills
        """
        return self.match_skills(resume_text, job_text)['missing_skills']
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _extract_by_dictionary(self, text: str, skill_dict: Dict[str, str]) -> List[Dict]:
        """
        Step 1: Exact dictionary matching.
        
        Fast but only catches exact matches (case-insensitive).
        Confidence: 1.0 (exact match)
        """
        found = []
        for skill_key, canonical_name in skill_dict.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill_key) + r'\b'
            if re.search(pattern, text):
                found.append({
                    'name': canonical_name,
                    'confidence': 1.0,
                    'method': 'dictionary',
                })
        
        return found
    
    def _extract_by_fuzzy_matching(self, text: str, skills: Set[str]) -> List[Dict]:
        """
        Step 2: Fuzzy matching to catch typos/abbreviations.
        
        Example: "pyton" → "python" (fuzzy match at 80% confidence)
        More flexible than exact matching but slower.
        """
        found = []
        text_words = set(re.findall(r'\b\w+\b', text))
        
        for skill in skills:
            for word in text_words:
                # Skip very short words (avoid noise)
                if len(word) < 3:
                    continue
                
                # Fuzzy match
                similarity = fuzz.ratio(word.lower(), skill.lower())
                if similarity >= self.fuzzy_threshold:
                    found.append({
                        'name': skill,
                        'confidence': similarity / 100.0,
                        'method': 'fuzzy',
                    })
        
        return found
    
    def _extract_by_tfidf(self, text: str, skills: Set[str]) -> List[Dict]:
        """
        Step 3: TF-IDF fallback (catch skills not in dictionary).
        
        Simple implementation: extract high-frequency tokens that MIGHT be skills.
        In production, could use sklearn's TfidfVectorizer for more sophistication.
        """
        # For this demo, we'll keep it simple: just report frequent words
        # that don't match stopwords.
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Filter for potentially-skill-like words (longer, frequent)
        found = []
        for word, freq in word_freq.most_common(20):
            if len(word) >= 4 and freq >= self.min_keyword_frequency:
                # Check if it LOOKS like a skill (camelCase, all caps, etc.)
                if self._looks_like_skill_word(word):
                    found.append({
                        'name': word,
                        'confidence': min(0.6, freq / 10),  # Low confidence for unknown skills
                        'method': 'tfidf',
                    })
        
        return found
    
    def _looks_like_skill_word(self, word: str) -> bool:
        """Heuristic: does word look like it could be a skill?"""
        # Contains numbers or special chars (C++, C#, AWS, etc.)
        if any(c.isdigit() or c in "#+@" for c in word):
            return True
        # Very long word (likely technical jargon)
        if len(word) > 10:
            return True
        # Already in our skill lists
        if word in self.hard_skills_lower or word in self.soft_skills_lower:
            return True
        return False
    
    def _merge_skill_results(self, skill_list: List[Dict]) -> List[Dict]:
        """
        Merge duplicate skills, keeping highest confidence.
        
        Example: [{'name': 'Python', 'conf': 1.0}, {'name': 'Python', 'conf': 0.8}]
                 → [{'name': 'Python', 'conf': 1.0}]
        """
        skill_dict = {}
        for skill in skill_list:
            name_lower = skill['name'].lower()
            if name_lower not in skill_dict or skill['confidence'] > skill_dict[name_lower]['confidence']:
                skill_dict[name_lower] = skill
        
        # Return sorted by confidence (descending)
        return sorted(skill_dict.values(), key=lambda s: s['confidence'], reverse=True)
    
    def add_custom_skills(self, hard_skills: Optional[Set[str]] = None, 
                         soft_skills: Optional[Set[str]] = None):
        """
        Add custom skills to the extractor (for domain-specific vocabularies).
        
        Example:
            extractor.add_custom_skills(
                hard_skills={'CUDA', 'OpenGL', 'MATLAB'},
                soft_skills={'Customer Success', 'Negotiation'}
            )
        """
        if hard_skills:
            self.hard_skills.update(hard_skills)
            self.hard_skills_lower.update({s.lower(): s for s in hard_skills})
        
        if soft_skills:
            self.soft_skills.update(soft_skills)
            self.soft_skills_lower.update({s.lower(): s for s in soft_skills})
        
        logger.info(f"✓ Added custom skills - Total hard: {len(self.hard_skills)}, soft: {len(self.soft_skills)}")


# Validation & Testing
def validate_skill_extraction(
    sample_resume: str,
    sample_job: str,
    expected_matched_skills: Set[str]
) -> bool:
    """
    Validate skill extraction against expected results.
    
    Returns:
        True if extraction matches expectations
    """
    extractor = SkillExtractor()
    matches = extractor.match_skills(sample_resume, sample_job)
    
    matched_set = set(matches['matched_skills'])
    expected_set = {s.lower() for s in expected_matched_skills}
    
    correct = matched_set & expected_set
    false_positives = matched_set - expected_set
    false_negatives = expected_set - matched_set
    
    logger.info(f"Skill Extraction Validation:")
    logger.info(f"  Correct: {correct}")
    logger.info(f"  False Positives: {false_positives}")
    logger.info(f"  False Negatives: {false_negatives}")
    
    return len(false_negatives) == 0


if __name__ == "__main__":
    # Test the skill extractor
    extractor = SkillExtractor()
    
    test_resume = """
    Senior Python Developer with 5 years of ML experience.
    Expertise: Python, Machine Learning, TensorFlow, PyTorch, AWS
    Led team in building scalable systems with Docker and Kubernetes.
    """
    
    test_job = """
    We're hiring a Senior ML Engineer.
    Required: Python, Machine Learning, AWS, Docker
    Preferred: TensorFlow, Kubernetes, Scala
    """
    
    print("Testing Skill Extraction:")
    print("="*60)
    
    resume_skills = extractor.extract_from_text(test_resume)
    print(f"\nResume Skills:")
    print(f"  Hard: {[s['name'] for s in resume_skills['hard_skills']]}")
    print(f"  Soft: {[s['name'] for s in resume_skills['soft_skills']]}")
    
    job_skills = extractor.extract_from_text(test_job)
    print(f"\nJob Skills:")
    print(f"  Hard: {[s['name'] for s in job_skills['hard_skills']]}")
    print(f"  Soft: {[s['name'] for s in job_skills['soft_skills']]}")
    
    matches = extractor.match_skills(test_resume, test_job)
    print(f"\nMatching Results:")
    print(f"  Matched: {matches['matched_skills']}")
    print(f"  Missing: {matches['missing_skills']}")
    print(f"  Match Score: {matches['match_score']:.2%}")
