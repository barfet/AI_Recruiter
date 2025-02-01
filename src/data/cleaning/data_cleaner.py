"""Main module for cleaning and enhancing candidate data"""

import json
from typing import Dict, Any
from pathlib import Path

from src.core.logging import setup_logger
from src.data.cleaning.education_cleaner import clean_education
from src.data.cleaning.skills_cleaner import clean_skills
from src.data.cleaning.location_cleaner import clean_location
from src.data.cleaning.industry_cleaner import clean_industry

logger = setup_logger(__name__)


def clean_candidate_data(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and enhance a single candidate profile"""
    try:
        # Extract experience text for skill extraction
        experience_text = " ".join(
            [
                f"{exp.get('title', '')} {exp.get('company', '')} {exp.get('description', '')}"
                for exp in candidate.get("experience", [])
            ]
        )

        # Clean education
        education = clean_education(candidate.get("education", []))

        # Clean skills
        skills = clean_skills(
            candidate.get("skills", []), experience_text=experience_text
        )

        # Clean location
        location = clean_location(candidate.get("location"))

        # Clean industry
        industries = clean_industry(
            candidate.get("industry"), skills=skills, experience_text=experience_text
        )

        # Update the candidate data
        cleaned_candidate = {
            **candidate,
            "education": [edu.dict() for edu in education],
            "skills": skills,
            "location": location,
            "industries": industries,  # New field with multiple possible industries
            "industry": (
                industries[0] if industries else "Information Technology"
            ),  # Keep primary industry for compatibility
        }

        return cleaned_candidate

    except Exception as e:
        logger.error(f"Error cleaning candidate data: {str(e)}")
        return candidate


def clean_candidates_file(
    input_file: Path, output_file: Path, batch_size: int = 100
) -> None:
    """Clean and enhance all candidate data"""
    try:
        # Load candidates
        with open(input_file, "r") as f:
            candidates = json.load(f)

        total_candidates = len(candidates)
        cleaned_candidates = []

        # Process in batches
        for i in range(0, total_candidates, batch_size):
            batch = candidates[i:i + batch_size]
            cleaned_batch = [clean_candidate_data(candidate) for candidate in batch]
            cleaned_candidates.extend(cleaned_batch)

            logger.info(
                f"Processed {min(i + batch_size, total_candidates)}/{total_candidates} candidates"
            )

        # Save cleaned data
        with open(output_file, "w") as f:
            json.dump(cleaned_candidates, f, indent=2)

        logger.info(
            f"Successfully cleaned and saved {total_candidates} candidate profiles"
        )

    except Exception as e:
        logger.error(f"Error processing candidates file: {str(e)}")
        raise
