"""Module for cleaning and normalizing education data"""

from typing import List, Dict, Any, Optional
import re
from src.data.models.candidate import Education

# Common degree patterns
DEGREE_PATTERNS = {
    "bachelor": [
        r"bachelor(?:'s)?(?: degree)?",
        r"b\.?s\.?",
        r"b\.?a\.?",
        r"b\.?e\.?",
        r"bachelor of (?:science|arts|engineering|business|technology)",
    ],
    "master": [
        r"master(?:'s)?(?: degree)?",
        r"m\.?s\.?",
        r"m\.?a\.?",
        r"m\.?e\.?",
        r"master of (?:science|arts|engineering|business|technology)",
        r"mba",
    ],
    "phd": [
        r"ph\.?d\.?",
        r"doctorate",
        r"doctor of philosophy",
    ],
    "associate": [
        r"associate(?:'s)?(?: degree)?",
        r"a\.?s\.?",
        r"a\.?a\.?",
    ],
    "diploma": [
        r"diploma",
        r"certification",
        r"certificate",
    ],
}

# Common fields of study
FIELDS_OF_STUDY = {
    "computer_science": [
        r"computer science",
        r"computing",
        r"software engineering",
        r"information technology",
        r"information systems",
    ],
    "engineering": [
        r"engineering",
        r"electrical",
        r"mechanical",
        r"civil",
    ],
    "business": [
        r"business",
        r"management",
        r"administration",
        r"finance",
        r"economics",
    ],
    "science": [
        r"science",
        r"physics",
        r"chemistry",
        r"biology",
        r"mathematics",
    ],
}


def extract_degree_level(text: str) -> Optional[str]:
    """Extract standardized degree level from text"""
    text = text.lower().strip()

    for level, patterns in DEGREE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return level.title()

    return None


def extract_field_of_study(text: str) -> Optional[str]:
    """Extract field of study from text"""
    text = text.lower().strip()

    for field, patterns in FIELDS_OF_STUDY.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return field.replace("_", " ").title()

    return None


def extract_institution(text: str) -> Optional[str]:
    """Extract institution name from text"""
    # Common patterns for institution names
    university_patterns = [
        r"university of [\w\s]+",
        r"[\w\s]+ university",
        r"[\w\s]+ college",
        r"[\w\s]+ institute of technology",
    ]

    text = text.lower().strip()

    for pattern in university_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).title()

    return None


def clean_education(education: List[Dict[str, Any]]) -> List[Education]:
    """Clean and normalize education entries"""
    cleaned = []
    seen = set()  # Track unique combinations to avoid duplicates

    for edu in education:
        # Combine degree and institution text for better extraction
        full_text = f"{edu.get('degree', '')} {edu.get('institution', '')}"

        # Extract components
        degree = extract_degree_level(full_text)
        field = extract_field_of_study(full_text)
        institution = extract_institution(full_text)

        if degree:  # Only keep entries where we could identify a degree
            # Create a unique key to detect duplicates
            key = f"{degree}-{field}-{institution}"
            if key not in seen:
                seen.add(key)

                # Combine degree and field if both exist
                full_degree = f"{degree} in {field}" if field else degree

                cleaned.append(
                    Education(
                        degree=full_degree,
                        institution=institution or "",
                        start_date=edu.get("start_date"),
                        end_date=edu.get("end_date"),
                    )
                )

    return cleaned
