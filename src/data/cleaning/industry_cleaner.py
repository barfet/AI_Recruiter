"""Module for cleaning and normalizing industry data"""

from typing import List, Optional, Set
import re

# Industry categories and their related terms
INDUSTRY_PATTERNS = {
    "Software Development": [
        r"software",
        r"web development",
        r"application development",
        r"programming",
        r"devops",
        r"full[\s-]stack",
    ],
    "Data Science & Analytics": [
        r"data science",
        r"machine learning",
        r"artificial intelligence",
        r"analytics",
        r"big data",
        r"statistics",
    ],
    "Cloud Computing": [
        r"cloud",
        r"aws",
        r"azure",
        r"google cloud",
        r"devops",
        r"infrastructure",
    ],
    "Cybersecurity": [
        r"security",
        r"cyber",
        r"information security",
        r"network security",
        r"cryptography",
    ],
    "IT Infrastructure": [
        r"infrastructure",
        r"system admin",
        r"network",
        r"it support",
        r"technical support",
    ],
    "Product Management": [
        r"product management",
        r"product owner",
        r"scrum master",
        r"agile",
        r"project management",
    ],
    "UX/UI Design": [
        r"ux",
        r"ui",
        r"user experience",
        r"user interface",
        r"web design",
        r"graphic design",
    ],
    "Quality Assurance": [
        r"qa",
        r"quality assurance",
        r"testing",
        r"test automation",
        r"quality engineer",
    ],
    "Database Administration": [
        r"database",
        r"dba",
        r"sql",
        r"data engineer",
        r"data architect",
    ],
    "Mobile Development": [
        r"mobile",
        r"ios",
        r"android",
        r"react native",
        r"flutter",
    ],
}

# Skill to industry mappings
SKILL_TO_INDUSTRY = {
    # Programming Languages -> Software Development
    "Java": "Software Development",
    "JavaScript": "Software Development",
    "TypeScript": "Software Development",
    "C++": "Software Development",
    "C#": "Software Development",
    "Go": "Software Development",
    # Data Science & ML
    "Python": [
        "Software Development",
        "Data Science & Analytics",
    ],  # Python is used in both
    "R": "Data Science & Analytics",
    "TensorFlow": "Data Science & Analytics",
    "PyTorch": "Data Science & Analytics",
    "Machine Learning": "Data Science & Analytics",
    "Deep Learning": "Data Science & Analytics",
    # Cloud & DevOps
    "AWS": "Cloud Computing",
    "Azure": "Cloud Computing",
    "Google Cloud": "Cloud Computing",
    "Kubernetes": "Cloud Computing",
    "Docker": "Cloud Computing",
    "Terraform": "Cloud Computing",
    # Security
    "Security": "Cybersecurity",
    "Penetration Testing": "Cybersecurity",
    "Encryption": "Cybersecurity",
    # Mobile
    "iOS": "Mobile Development",
    "Android": "Mobile Development",
    "Swift": "Mobile Development",
    "Kotlin": "Mobile Development",
    "React Native": "Mobile Development",
    # Design
    "UI": "UX/UI Design",
    "UX": "UX/UI Design",
    "Figma": "UX/UI Design",
    "Adobe XD": "UX/UI Design",
}


def determine_industry_from_skills(skills: List[str]) -> Set[str]:
    """Determine possible industries based on skills"""
    industries = set()

    for skill in skills:
        if skill in SKILL_TO_INDUSTRY:
            value = SKILL_TO_INDUSTRY[skill]
            if isinstance(value, list):
                industries.update(value)
            else:
                industries.add(value)

    return industries


def determine_industry_from_text(text: str) -> Set[str]:
    """Determine possible industries based on text description"""
    industries = set()
    text = text.lower()

    for industry, patterns in INDUSTRY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                industries.add(industry)
                break

    return industries


def clean_industry(
    current_industry: Optional[str], skills: List[str] = None, experience_text: str = ""
) -> List[str]:
    """Clean and enhance industry classification"""
    industries = set()

    # Start with current industry if it's valid
    if current_industry:
        current_industry = current_industry.strip().title()
        # Map some common variations
        if "IT" in current_industry.upper() or "TECH" in current_industry.upper():
            industries.add("Software Development")
        elif "DATA" in current_industry.upper():
            industries.add("Data Science & Analytics")
        elif "SECURITY" in current_industry.upper():
            industries.add("Cybersecurity")

    # Add industries based on skills
    if skills:
        industries.update(determine_industry_from_skills(skills))

    # Add industries based on experience description
    if experience_text:
        industries.update(determine_industry_from_text(experience_text))

    # If no industries found, use a default
    if not industries and current_industry:
        industries.add(current_industry)
    elif not industries:
        industries.add("Information Technology")

    return sorted(list(industries))
