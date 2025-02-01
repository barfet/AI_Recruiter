"""Module for cleaning and normalizing skills data"""

from typing import List, Set
import re

# Common skill variations and their normalized forms
SKILL_MAPPINGS = {
    # Programming Languages
    r"javascript|js": "JavaScript",
    r"typescript|ts": "TypeScript",
    r"python|py": "Python",
    r"java\b": "Java",
    r"c\+\+|cpp": "C++",
    r"c#|csharp": "C#",
    r"golang|go\b": "Go",
    r"ruby\b": "Ruby",
    r"php\b": "PHP",
    r"rust\b": "Rust",
    r"scala\b": "Scala",
    r"kotlin\b": "Kotlin",
    r"swift\b": "Swift",
    # Web Technologies
    r"react\.?js|react\b": "React",
    r"vue\.?js|vue\b": "Vue.js",
    r"angular(?:\.js)?|ng": "Angular",
    r"node\.?js": "Node.js",
    r"express\.?js": "Express.js",
    r"next\.?js": "Next.js",
    r"html5?": "HTML",
    r"css3?": "CSS",
    r"sass|scss": "SASS",
    r"webpack": "Webpack",
    # Databases
    r"mysql": "MySQL",
    r"postgresql|postgres": "PostgreSQL",
    r"mongodb|mongo": "MongoDB",
    r"redis": "Redis",
    r"elasticsearch|elastic": "Elasticsearch",
    r"cassandra": "Cassandra",
    r"sql\b": "SQL",
    # Cloud & DevOps
    r"aws|amazon web services": "AWS",
    r"gcp|google cloud": "Google Cloud",
    r"azure": "Azure",
    r"docker": "Docker",
    r"kubernetes|k8s": "Kubernetes",
    r"terraform": "Terraform",
    r"jenkins": "Jenkins",
    r"git\b": "Git",
    # AI/ML
    r"tensorflow|tf": "TensorFlow",
    r"pytorch": "PyTorch",
    r"scikit[\s-]learn": "Scikit-learn",
    r"pandas": "Pandas",
    r"numpy": "NumPy",
    r"machine learning|ml": "Machine Learning",
    r"deep learning|dl": "Deep Learning",
    r"nlp|natural language processing": "NLP",
    # Methodologies
    r"agile": "Agile",
    r"scrum": "Scrum",
    r"kanban": "Kanban",
    r"tdd|test driven development": "TDD",
    r"ci/cd|cicd|continuous integration": "CI/CD",
    # Testing
    r"junit": "JUnit",
    r"jest": "Jest",
    r"selenium": "Selenium",
    r"cypress": "Cypress",
    r"pytest": "PyTest",
}


def normalize_skill(skill: str) -> str:
    """Normalize a single skill"""
    skill = skill.strip().lower()

    for pattern, normalized in SKILL_MAPPINGS.items():
        if re.match(f"^{pattern}$", skill, re.IGNORECASE):
            return normalized

    # If no match found, capitalize words
    return " ".join(word.capitalize() for word in skill.split())


def extract_skills_from_text(text: str) -> Set[str]:
    """Extract skills from text using pattern matching"""
    found_skills = set()

    # Convert text to lowercase for matching
    text = text.lower()

    # Look for skill patterns in the text
    for pattern, normalized in SKILL_MAPPINGS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found_skills.add(normalized)

    return found_skills


def clean_skills(skills: List[str], experience_text: str = "") -> List[str]:
    """Clean and normalize skills list, optionally extracting additional skills from experience"""
    # Normalize existing skills
    normalized_skills = {normalize_skill(skill) for skill in skills if skill.strip()}

    # Extract additional skills from experience text if provided
    if experience_text:
        additional_skills = extract_skills_from_text(experience_text)
        normalized_skills.update(additional_skills)

    # Sort for consistent ordering
    return sorted(list(normalized_skills))
