"""
Train and save the resume-job matching model.

This script creates synthetic labeled resume/job pairs, trains a TF-IDF plus
Logistic Regression pipeline, evaluates it, and saves:
    models/1.0.0/pipeline.joblib
    models/1.0.0/metadata.json

Install optional ML dependencies first:
    pip install -r requirements-ml.txt

Run:
    python -m project.train_model
"""

from __future__ import annotations

import json
import random
from datetime import datetime, UTC
from pathlib import Path

import joblib

from .data_generation import SOFT_SKILLS, TECHNICAL_SKILLS


def main() -> None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependencies. Run: pip install -r requirements-ml.txt"
        ) from exc

    texts, y = build_training_pairs(sample_count=1800, seed=42)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }

    model_dir = Path("models") / "1.0.0"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "pipeline.joblib")

    metadata = {
        "model_id": "resume_matcher_tfidf_logreg",
        "version": "1.0.0",
        "model_type": "tfidf_logistic_regression",
        "training_date": datetime.now(UTC).isoformat(timespec="seconds"),
        "main_metric": metrics,
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "feature_count": len(pipeline.named_steps["tfidf"].get_feature_names_out()),
        "hyperparameters": {
            "tfidf_max_features": 6000,
            "ngram_range": [1, 2],
            "classifier": "LogisticRegression(class_weight='balanced')",
        },
        "notes": "Synthetic data model for portfolio demonstration. Replace with representative labeled data for production.",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved model to models/1.0.0/pipeline.joblib")
    print(json.dumps(metrics, indent=2))


def build_training_pairs(sample_count: int = 1800, seed: int = 42):
    """
    Build balanced synthetic resume-job pairs with controlled skill overlap.

    Positive pairs share most required skills and compatible seniority.
    Negative pairs intentionally mismatch core skills and seniority.
    """
    random.seed(seed)
    all_skills = [skill for skills in TECHNICAL_SKILLS.values() for skill in skills]
    levels = ["entry", "junior", "mid", "senior"]
    texts = []
    labels = []

    for index in range(sample_count):
        is_match = index % 2 == 0
        level = random.choice(levels)
        resume_skills = set(random.sample(all_skills, 8))
        soft = set(random.sample(SOFT_SKILLS, 3))

        if is_match:
            required = set(random.sample(list(resume_skills), 5))
            preferred = set(random.sample(list((set(all_skills) - required) | resume_skills), 3))
            job_level = level
        else:
            required_pool = list(set(all_skills) - resume_skills)
            required = set(random.sample(required_pool, 5))
            preferred = set(random.sample(required_pool, 3))
            job_level = random.choice([item for item in levels if item != level])

        resume = render_resume(level, sorted(resume_skills | soft))
        job = render_job(job_level, sorted(required), sorted(preferred))
        texts.append(f"RESUME:\n{resume}\n\nJOB:\n{job}")
        labels.append(1 if is_match else 0)

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    shuffled_texts, shuffled_labels = zip(*combined)
    return list(shuffled_texts), list(shuffled_labels)


def render_resume(level: str, skills):
    years = {"entry": 1, "junior": 3, "mid": 6, "senior": 10}[level]
    return f"""
    Professional Summary
    {level.title()} software and data professional with {years} years of experience.

    Skills
    {", ".join(skills)}

    Experience
    Built production systems using {skills[0]}, {skills[1]}, and {skills[2]}.
    Delivered data products, APIs, automation, and stakeholder-facing reports.
    Collaborated with teams and explained technical decisions to business users.
    """


def render_job(level: str, required, preferred):
    years = {"entry": 0, "junior": 2, "mid": 5, "senior": 8}[level]
    return f"""
    Job Description
    We are hiring a {level.title()} engineer for an AI and software delivery team.

    Requirements
    {years}+ years of experience.
    Required skills: {", ".join(required)}.
    Strong communication, collaboration, and problem solving.

    Preferred
    Nice to have: {", ".join(preferred)}.
    Experience deploying reliable production systems.
    """


if __name__ == "__main__":
    main()
