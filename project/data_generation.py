"""
Synthetic data generation for resume screening.

This module provides tools to:
1. Create realistic synthetic resumes
2. Create realistic job descriptions
3. Generate matching labels
4. Load and combine public datasets (when available)
"""

import json
import random
from typing import List, Tuple, Dict
from datetime import datetime
from pathlib import Path


# Technical Skills Library
TECHNICAL_SKILLS = {
    "Languages": ["Python", "Java", "C++", "JavaScript", "Go", "Rust", "Ruby", "PHP", "Kotlin", "R"],
    "MLOps": ["TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy", "Hugging Face", "MLflow", "Experiment Tracking"],
    "Data": ["SQL", "Spark", "Hadoop", "Cassandra", "MongoDB", "PostgreSQL", "Redis", "Data Warehousing"],
    "Backend": ["FastAPI", "Django", "Spring Boot", "Node.js", "Flask", "Express", "ASP.NET"],
    "DevOps": ["Docker", "Kubernetes", "CI/CD", "GitHub Actions", "Jenkins", "AWS", "Azure", "GCP"],
    "Frontend": ["React", "Vue.js", "Angular", "TypeScript", "Tailwind CSS", "Streamlit"],
}

# Soft Skills Library
SOFT_SKILLS = [
    "Leadership", "Project Management", "Communication", "Teamwork", "Problem Solving",
    "Adaptability", "Attention to Detail", "Critical Thinking", "Mentoring",
]

# Experience bullet templates
EXPERIENCE_TEMPLATES = [
    "Built {} system that improved {} by {}%",
    "Led team of {} engineers to deliver {}",
    "Developed {} solution that reduced {} costs by {}%",
    "Collaborated with {} on {} project",
    "Optimized {} algorithm achieving {}x speedup",
]

# Company names
COMPANIES = [
    "TechCorp", "DataSystems Inc", "CloudFirst Ltd", "AI Innovations", "Digital Solutions",
    "Tech Startup", "Enterprise Software Co", "Big Tech Corp", "FinTech Solutions"
]

# Job titles
JOB_TITLES = {
    "entry": ["Junior Data Scientist", "Associate ML Engineer", "Graduate Software Engineer"],
    "junior": ["Data Scientist", "ML Engineer", "Software Engineer"],
    "mid": ["Senior Data Scientist", "Senior ML Engineer", "Senior Software Engineer", "ML Architect"],
    "senior": ["Principal ML Engineer", "Staff Data Scientist", "ML Team Lead"],
}


class SyntheticDataGenerator:
    """Generate synthetic resumes and job descriptions for testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        random.seed(seed)
        self.resume_counter = 0
        self.job_counter = 0
    
    def generate_resume(
        self,
        experience_level: str = "mid",
        num_skills: int = None,
        include_soft_skills: bool = True,
    ) -> Dict:
        """
        Generate a synthetic resume.
        
        Args:
            experience_level: "entry", "junior", "mid", "senior"
            num_skills: Number of technical skills to include. If None, auto-select based on level.
            include_soft_skills: Whether to mention soft skills
            
        Returns:
            Dictionary with resume data
        """
        self.resume_counter += 1
        resume_id = f"R{self.resume_counter:05d}"
        
        # Experience mapping
        experience_ranges = {"entry": (0, 2), "junior": (2, 4), "mid": (4, 8), "senior": (8, 15)}
        years = random.uniform(*experience_ranges[experience_level])
        
        # Skill count mapping
        if num_skills is None:
            num_skills = {"entry": 3, "junior": 5, "mid": 8, "senior": 10}.get(experience_level, 5)
        
        # Select skills
        all_skills = []
        for category in TECHNICAL_SKILLS.values():
            all_skills.extend(category)
        skills = random.sample(all_skills, min(num_skills, len(all_skills)))
        
        if include_soft_skills:
            skills.extend(random.sample(SOFT_SKILLS, min(3, len(SOFT_SKILLS))))
        
        # Build resume text
        company = random.choice(COMPANIES)
        bullet_count = {"entry": 2, "junior": 3, "mid": 4, "senior": 5}.get(experience_level, 3)
        experience_bullets = self._generate_experience_bullets(skills, bullet_count)
        
        resume_text = f"""
        RESUME - {resume_id}
        
        PROFESSIONAL SUMMARY
        {experience_level.title()} professional with {years:.0f} years of experience in software engineering and data science.
        Expertise in building scalable systems and deploying machine learning models.
        
        EXPERIENCE
        Technical Lead at {company}
        {experience_bullets}
        
        SKILLS
        {', '.join(skills)}
        
        EDUCATION
        B.S. Computer Science
        """
        
        return {
            "resume_id": resume_id,
            "candidate_name": f"Candidate {self.resume_counter}",
            "email": f"candidate{self.resume_counter}@example.com",
            "raw_text": resume_text,
            "experience_years": years,
            "experience_level": experience_level,
            "skills_mentioned": skills,
        }
    
    def generate_job_description(
        self,
        experience_level: str = "mid",
        num_required_skills: int = 5,
    ) -> Dict:
        """
        Generate a synthetic job description.
        
        Args:
            experience_level: "entry", "junior", "mid", "senior"
            num_required_skills: Number of required skills
            
        Returns:
            Dictionary with job data
        """
        self.job_counter += 1
        job_id = f"J{self.job_counter:05d}"
        
        # Select title and experience requirement
        title = random.choice(JOB_TITLES[experience_level])
        years_req = {"entry": 0, "junior": 2, "mid": 5, "senior": 8}.get(experience_level, 3)
        
        # Select required and preferred skills
        all_skills = []
        for category in TECHNICAL_SKILLS.values():
            all_skills.extend(category)
        
        required_skills = random.sample(all_skills, min(num_required_skills, len(all_skills)))
        preferred_skills = random.sample(
            [s for s in all_skills if s not in required_skills],
            min(3, len(all_skills) - len(required_skills))
        )
        
        jd_text = f"""
        JOB DESCRIPTION - {job_id}
        
        Title: {title}
        Company: {random.choice(COMPANIES)}
        
        ABOUT THE ROLE
        We are seeking a talented {title} to join our team. You will work on developing
        cutting-edge solutions using latest technologies.
        
        REQUIREMENTS
        - {years_req}+ years of professional experience
        - Strong background in: {', '.join(required_skills)}
        - Experience with system design and architecture
        - Excellent communication and teamwork skills
        
        PREFERRED QUALIFICATIONS
        - Experience with: {', '.join(preferred_skills)}
        - Published research or open source contributions
        - Experience scaling systems to production
        
        RESPONSIBILITIES
        - Design and implement scalable systems
        - Work with cross-functional teams
        - Mentor junior engineers
        - Drive technical innovation
        """
        
        return {
            "job_id": job_id,
            "title": title,
            "company": random.choice(COMPANIES),
            "raw_text": jd_text,
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "experience_years_required": years_req,
            "experience_level": experience_level,
        }
    
    def generate_match_labels(
        self,
        resumes: List[Dict],
        jobs: List[Dict],
        match_rate: float = 0.3,
    ) -> List[Dict]:
        """
        Generate matching labels for resume-job pairs.
        
        Strategy:
        1. Resumes with >= 70% skill overlap → match
        2. Experience level within 1 tier → match
        3. Otherwise → no match
        
        Args:
            resumes: List of resume dictionaries
            jobs: List of job description dictionaries
            match_rate: Target proportion of matches (for synthetic variety)
            
        Returns:
            List of match label dictionaries
        """
        labels = []
        experience_hierarchy = {"entry": 0, "junior": 1, "mid": 2, "senior": 3}
        
        for resume in resumes:
            for job in jobs:
                resume_skills = set(s.lower() for s in resume.get("skills_mentioned", []))
                job_skills = set(s.lower() for s in job.get("required_skills", []))
                
                # Calculate overlap
                if job_skills:
                    skill_overlap = len(resume_skills & job_skills) / len(job_skills)
                else:
                    skill_overlap = 0.0
                
                # Check experience level compatibility
                resume_level = experience_hierarchy.get(resume.get("experience_level", "mid"), 2)
                job_level = experience_hierarchy.get(job.get("experience_level", "mid"), 2)
                level_diff = abs(resume_level - job_level)
                
                # Matching logic
                is_match = (skill_overlap >= 0.6) and (level_diff <= 1)
                
                # Add randomness for variety
                if random.random() < 0.1:  # 10% noise
                    is_match = not is_match
                
                labels.append({
                    "resume_id": resume["resume_id"],
                    "job_id": job["job_id"],
                    "is_match": is_match,
                    "confidence": 0.95 if skill_overlap > 0.7 else 0.7,
                    "label_timestamp": datetime.utcnow().isoformat(),
                })
        
        return labels
    
    def _generate_experience_bullets(self, skills: List[str], count: int) -> str:
        """Generate realistic experience bullet points."""
        bullets = []
        for _ in range(count):
            skill = random.choice(skills)
            improvements = ["performance", "efficiency", "accuracy", "latency"]
            improvement = random.choice(improvements)
            percentage = random.randint(10, 50)
            
            bullet = f"• Developed {skill} solution that improved {improvement} by {percentage}%"
            bullets.append(bullet)
        
        return "\n".join(bullets)
    
    def save_dataset(self, output_dir: str = "data") -> Tuple[str, str, str]:
        """
        Generate and save a complete dataset.
        
        Args:
            output_dir: Directory to save data files
            
        Returns:
            Paths to resumes.json, jobs.json, labels.json
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate data
        resumes = [
            self.generate_resume(level, num_skills=random.randint(4, 12))
            for level in ["entry", "entry", "junior", "junior", "mid", "mid", "senior", "senior"]
        ]
        
        jobs = [
            self.generate_job_description(level)
            for level in ["entry", "junior", "mid", "senior"]
        ]
        
        labels = self.generate_match_labels(resumes, jobs)
        
        # Save
        resumes_path = output_path / "resumes.json"
        jobs_path = output_path / "jobs.json"
        labels_path = output_path / "labels.json"
        
        with open(resumes_path, "w") as f:
            json.dump(resumes, f, indent=2, default=str)
        
        with open(jobs_path, "w") as f:
            json.dump(jobs, f, indent=2, default=str)
        
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2, default=str)
        
        print(f"✓ Generated {len(resumes)} resumes → {resumes_path}")
        print(f"✓ Generated {len(jobs)} jobs → {jobs_path}")
        print(f"✓ Generated {len(labels)} labels → {labels_path}")
        
        return str(resumes_path), str(jobs_path), str(labels_path)


if __name__ == "__main__":
    # Example usage
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate sample resume
    resume = generator.generate_resume(experience_level="mid")
    print("Sample Resume:")
    print(json.dumps(resume, indent=2, default=str))
    print("\n" + "="*80 + "\n")
    
    # Generate sample job
    job = generator.generate_job_description(experience_level="mid")
    print("Sample Job Description:")
    print(json.dumps(job, indent=2, default=str))
    print("\n" + "="*80 + "\n")
    
    # Generate full dataset
    print("Generating full dataset...")
    generator = SyntheticDataGenerator(seed=42)
    resumes_file, jobs_file, labels_file = generator.save_dataset(output_dir="data")
