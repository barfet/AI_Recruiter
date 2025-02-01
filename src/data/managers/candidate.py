import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

from src.data.models.candidate import CandidateProfile
from src.data.managers.base import BaseManager
from src.core.config import settings
from src.core.logging import setup_logger
from src.data.cleaning.location_cleaner import clean_location
from src.data.cleaning.education_cleaner import clean_education
from src.data.cleaning.skills_cleaner import clean_skills, extract_skills_from_text
from src.data.cleaning.industry_cleaner import clean_industry

logger = setup_logger(__name__)


class CandidateManager(BaseManager):
    """Manager for candidate data"""

    def __init__(self):
        super().__init__(CandidateProfile)
        self.resume_dir = settings.RESUME_DIR

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                boxes_flow=0.5,
                detect_vertical=False,
                all_texts=True,
            )
            text = extract_text(str(pdf_path), laparams=laparams)
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def find_pdf_by_id(self, resume_id: str) -> Optional[Tuple[Path, str]]:
        """Find PDF file by resume ID"""
        for industry_dir in self.resume_dir.iterdir():
            if not industry_dir.is_dir():
                continue

            pdf_path = industry_dir / f"{resume_id}.pdf"
            if pdf_path.exists():
                return pdf_path, industry_dir.name
        return None, None

    def extract_contact_info(self, text: str) -> Dict:
        """Extract contact information from resume text"""

        contact_info = {
            "email": None,
            "phone": None,
            "location": None,
            "linkedin": None,
            "website": None,
        }

        # Email pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info["email"] = email_match.group()

        # Phone pattern (various formats)
        phone_pattern = r"""
            (?:                     # Start of non-capturing group
            (?:\+\d{1,2}\s*)?      # Optional country code
            \(?\d{3}\)?            # Area code with optional parentheses
            [\s.-]?                # Optional separator
            \d{3}                  # First three digits
            [\s.-]?                # Optional separator
            \d{4}                  # Last four digits
            )                      # End of non-capturing group
        """
        phone_match = re.search(phone_pattern, text, re.VERBOSE)
        if phone_match:
            contact_info["phone"] = phone_match.group()

        # Location pattern (City, State or City, Country)
        # pylint: disable=implicit-str-concat
        location_pattern = (
            r"(?:^|\n)"  # Start of line or newline
            r"(?:Address:|Location:|Based in:)?\s*"  # Optional label
            r"([A-Z][a-zA-Z\s.-]+,\s*[A-Z]{2,})"  # City, State format
        )
        # pylint: enable=implicit-str-concat
        location_match = re.search(location_pattern, text)
        if location_match:
            raw_location = location_match.group(1).strip()
            contact_info["location"] = clean_location(raw_location)

        # LinkedIn pattern
        linkedin_pattern = r"(?:linkedin\.com/in/|LinkedIn:?\s*)([a-zA-Z0-9-/]+)"
        linkedin_match = re.search(linkedin_pattern, text.lower())
        if linkedin_match:
            contact_info["linkedin"] = f"linkedin.com/in/{linkedin_match.group(1)}"

        # Website pattern
        website_pattern = (
            r"(?:Website:?\s*)?(?:https?://)?(?:www\.)?"
            r"([a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/\S*)?)"
        )
        website_match = re.search(website_pattern, text.lower())
        if website_match and "linkedin" not in website_match.group():
            contact_info["website"] = website_match.group()

        return contact_info

    def extract_education(self, text: str) -> List[Dict]:
        """Extract education information from resume text"""

        # First pass extraction using basic patterns
        education = []

        # Common education keywords
        edu_keywords = (
            r"(?:Bachelor|Master|PhD|B\.S\.|M\.S\.|M\.A\.|B\.A\.|" r"Associate|Diploma)"
        )
        degree_pattern = f"{edu_keywords}[\\s\\w]*(?:of|in|,)?\\s*[\\w\\s,]*"

        # Date pattern
        date_pattern = (
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)\s*\d{4}"
        )

        # Find education sections
        sections = re.split(r"\n{2,}", text)
        for section in sections:
            degree_match = re.search(degree_pattern, section, re.IGNORECASE)
            if degree_match:
                edu_entry = {
                    "degree": degree_match.group().strip(),
                    "institution": "",
                    "start_date": None,
                    "end_date": None,
                }

                # Try to find institution name
                lines = section.split("\n")
                for line in lines:
                    if any(x in line for x in ["University", "College", "Institute"]):
                        edu_entry["institution"] = line.strip()
                        break

                # Try to find dates
                dates = re.findall(date_pattern, section)
                if len(dates) >= 2:
                    edu_entry["start_date"] = dates[0]
                    edu_entry["end_date"] = dates[1]
                elif len(dates) == 1:
                    edu_entry["end_date"] = dates[0]

                education.append(edu_entry)

        # Clean and normalize the extracted education data
        return clean_education(education)

    def _get_date_pattern(self) -> str:
        """Get the regex pattern for dates"""
        return (
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)\s*\d{4}"
        )

    def _parse_dates(self, section: str, date_pattern: str) -> tuple[str | None, str | None]:
        """Parse start and end dates from a section"""
        dates = re.findall(date_pattern, section)
        start_date = dates[0] if dates else None
        end_date = None

        if len(dates) >= 2:
            end_date = dates[1]
        elif "present" in section.lower():
            end_date = "Present"

        return start_date, end_date

    def _parse_role(self, lines: list[str]) -> tuple[str, str]:
        """Parse job title and company from lines"""
        title_keywords = ["engineer", "developer", "manager", "analyst", "consultant"]
        title = ""
        company = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if not title and any(kw in line.lower() for kw in title_keywords):
                title = line
            elif not company and line != title:
                company = line
                break

        return title, company

    def _parse_description(self, lines: list[str], date_pattern: str) -> str:
        """Parse job description from lines"""
        desc_lines = []
        for line in lines[2:]:  # Skip title and company lines
            if line.strip() and not re.search(date_pattern, line):
                desc_lines.append(line.strip())
        return " ".join(desc_lines)

    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience from resume text"""
        experience = []
        date_pattern = self._get_date_pattern()
        sections = re.split(r"\n{2,}", text)

        for section in sections:
            if not re.search(date_pattern, section):
                continue

            lines = section.split("\n")
            start_date, end_date = self._parse_dates(section, date_pattern)
            title, company = self._parse_role(lines)
            description = self._parse_description(lines, date_pattern)

            if title or company:  # Only add if we found at least a title or company
                experience.append({
                    "title": title,
                    "company": company,
                    "start_date": start_date,
                    "end_date": end_date,
                    "description": description,
                })

        return experience

    def extract_skills(self, text: str, experience_text: str = "") -> List[str]:
        """Extract skills from resume text"""

        # Extract skills from both the main text and experience descriptions
        skills = extract_skills_from_text(text)
        if experience_text:
            skills.update(extract_skills_from_text(experience_text))

        # Clean and normalize the skills
        return clean_skills(list(skills))

    def process_candidates(self) -> List[CandidateProfile]:
        """Process candidate resumes from PDF files"""

        processed = []

        try:
            # Iterate through industry directories
            for industry_dir in self.resume_dir.iterdir():
                if not industry_dir.is_dir():
                    continue

                logger.info(f"Processing resumes in {industry_dir.name}")

                # Process each PDF in the directory
                for pdf_file in industry_dir.glob("*.pdf"):
                    try:
                        # Extract resume ID from filename
                        resume_id = pdf_file.stem
                        logger.info(f"Processing resume {resume_id}")

                        # Extract text from PDF
                        text = self.extract_text_from_pdf(pdf_file)
                        if not text:
                            continue

                        # Extract experience first to use in skills extraction
                        experience = self.extract_experience(text)
                        experience_text = " ".join(
                            [
                                f"{exp.get('title', '')} {exp.get('company', '')} "
                                f"{exp.get('description', '')}"
                                for exp in experience
                            ]
                        )

                        # Extract and clean all information
                        contact_info = self.extract_contact_info(text)
                        education = self.extract_education(text)
                        skills = self.extract_skills(text, experience_text)

                        # Clean industry using all available information
                        industries = clean_industry(
                            industry_dir.name,
                            skills=skills,
                            experience_text=experience_text,
                        )

                        # Create candidate profile with cleaned data
                        profile = CandidateProfile(
                            resume_id=resume_id,
                            email=contact_info["email"],
                            phone=contact_info["phone"],
                            location=contact_info["location"],
                            linkedin_url=contact_info["linkedin"],
                            website=contact_info["website"],
                            education=education,
                            experience=experience,
                            skills=skills,
                            industry=industries[0],  # Primary industry
                            industries=industries,  # All relevant industries
                            name="",  # Anonymized
                            summary="",
                            languages=[],
                            certifications=[],
                            desired_role=None,
                            desired_salary=None,
                            desired_location=None,
                            remote_preference=None,
                            github_url=None,
                            portfolio_url=None,
                        )

                        processed.append(profile)

                    except Exception as e:
                        logger.error(f"Error processing {pdf_file}: {str(e)}")
                        continue

            return processed

        except Exception as e:
            logger.error(f"Error processing candidates: {str(e)}")
            return []
