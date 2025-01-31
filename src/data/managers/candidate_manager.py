from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import re
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

from src.data.models.candidate import CandidateProfile, Education, Experience
from src.data.managers.base import BaseDataManager
from src.core.logging import setup_logger
from src.core.exceptions import DataIngestionError

logger = setup_logger(__name__)

class CandidateManager(BaseDataManager):
    """Manager for candidate profile data"""
    
    def __init__(self):
        super().__init__(CandidateProfile)
        self.raw_resumes_dir = self.raw_data_dir / "resumes/data/data"
        self.processed_candidates_file = self.processed_data_dir / "structured_candidates.json"
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using pdfminer"""
        try:
            return extract_text(str(pdf_path), laparams=LAParams())
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return ""
            
    def find_pdf_by_id(self, resume_id: str) -> Tuple[Optional[Path], Optional[str]]:
        """Find PDF file by resume ID across all industry folders"""
        # Search in all industry folders
        for industry_dir in self.raw_resumes_dir.glob("*"):
            if not industry_dir.is_dir():
                continue
                
            pdf_path = industry_dir / f"{resume_id}.pdf"
            if pdf_path.exists():
                return pdf_path, industry_dir.name
        
        return None, None
        
    def extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact information from resume text"""
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        # Phone pattern (various formats)
        phone_pattern = r'''(?:
            (?:\+\d{1,2}\s*)?                    # Optional country code
            \(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}  # (123) 456-7890 or 123-456-7890
        )'''
        phones = re.findall(phone_pattern, text, re.VERBOSE)
        
        # Location pattern (City, State or City, Country)
        location_pattern = r'(?:^|\n)(?:Address:|Location:|Based in:)?\s*([A-Z][a-zA-Z\s.-]+,\s*[A-Z]{2,})'
        locations = re.findall(location_pattern, text)
        
        # LinkedIn pattern
        linkedin_pattern = r'(?:linkedin\.com/in/|LinkedIn:?\s*)([a-zA-Z0-9-/]+)'
        linkedin = re.findall(linkedin_pattern, text.lower())
        
        # Website pattern
        website_pattern = r'(?:Website:?\s*)?(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/\S*)?)'
        websites = re.findall(website_pattern, text.lower())
        
        return {
            'email': emails[0] if emails else None,
            'phone': phones[0] if phones else None,
            'location': locations[0] if locations else None,
            'linkedin_url': f"linkedin.com/in/{linkedin[0]}" if linkedin else None,
            'portfolio_url': websites[0] if websites and 'linkedin' not in websites[0] else None
        }
        
    def extract_education(self, text: str) -> List[Dict[str, Any]]:
        """Extract education information from resume text"""
        education_entries = []
        
        # Common degree terms
        degree_terms = [
            'Bachelor', 'BS', 'BA', 'B.S.', 'B.A.',
            'Master', 'MS', 'MA', 'M.S.', 'M.A.',
            'PhD', 'Ph.D.', 'Doctorate',
            'Associate', 'AA', 'A.A.',
            'Certificate', 'Certification'
        ]
        
        # Split text into sections
        sections = text.split('\n\n')
        for section in sections:
            if any(term in section for term in degree_terms):
                try:
                    # Extract degree and field
                    degree_match = re.search(r'(' + '|'.join(degree_terms) + r'[^,\n]*)', section)
                    degree = degree_match.group(1) if degree_match else ''
                    
                    # Extract institution
                    institution_match = re.search(r'(?:from |at |- )(.*?)(?:,|\n|$)', section)
                    institution = institution_match.group(1) if institution_match else ''
                    
                    # Extract dates
                    date_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}'
                    dates = re.findall(date_pattern, section)
                    
                    if degree and institution:
                        education_entries.append({
                            'institution': institution.strip(),
                            'degree': degree.strip(),
                            'field_of_study': '',  # Would need more sophisticated parsing
                            'start_date': dates[0] if len(dates) > 0 else None,
                            'end_date': dates[-1] if len(dates) > 1 else None,
                            'description': section.strip()
                        })
                except Exception as e:
                    logger.warning(f"Error extracting education entry: {str(e)}")
                    continue
        
        return education_entries
        
    def extract_experience(self, text: str) -> List[Dict[str, Any]]:
        """Extract work experience information from resume text"""
        experience_entries = []
        
        # Split text into sections
        sections = text.split('\n\n')
        for section in sections:
            if len(section.strip()) > 50:  # Skip short sections
                try:
                    # Extract title and company
                    first_line = section.split('\n')[0]
                    title_company = first_line.split(' at ')
                    
                    if len(title_company) == 2:
                        title = title_company[0].strip()
                        company = title_company[1].strip()
                        
                        # Extract dates
                        date_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}'
                        dates = re.findall(date_pattern, section)
                        
                        # Extract location
                        location_match = re.search(r'([A-Z][a-zA-Z\s.-]+,\s*[A-Z]{2,})', section)
                        location = location_match.group(1) if location_match else None
                        
                        # Extract skills (look for technical terms)
                        skill_pattern = r'\b(?:Python|Java|C\+\+|JavaScript|SQL|AWS|Azure|React|Node\.js|Docker|Kubernetes|Machine Learning|AI|DevOps|Agile|Scrum)\b'
                        skills = list(set(re.findall(skill_pattern, section)))
                        
                        experience_entries.append({
                            'title': title,
                            'company': company,
                            'start_date': dates[0] if len(dates) > 0 else None,
                            'end_date': dates[-1] if len(dates) > 1 else None,
                            'description': section.strip(),
                            'location': location,
                            'skills': skills
                        })
                except Exception as e:
                    logger.warning(f"Error extracting experience entry: {str(e)}")
                    continue
        
        return experience_entries
        
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        # Common technical skills
        skill_patterns = [
            r'\b(?:Python|Java|C\+\+|JavaScript|TypeScript|Ruby|PHP|Swift|Kotlin|Go|Rust)\b',
            r'\b(?:SQL|MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle|SQLite)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|Linux|Unix|Windows)\b',
            r'\b(?:React|Angular|Vue|Node\.js|Express|Django|Flask|Spring|Laravel)\b',
            r'\b(?:Machine Learning|Deep Learning|AI|NLP|Computer Vision|Data Science)\b',
            r'\b(?:HTML|CSS|SASS|LESS|Bootstrap|Tailwind|Material-UI|jQuery)\b',
            r'\b(?:DevOps|Agile|Scrum|CI/CD|TDD|REST|GraphQL|Microservices)\b'
        ]
        
        skills = set()
        for pattern in skill_patterns:
            matches = re.findall(pattern, text)
            skills.update(matches)
        
        return list(skills)
        
    def load_and_process_candidates(self) -> List[CandidateProfile]:
        """Load raw candidate data, process it, and return validated candidate profiles"""
        try:
            processed_candidates = []
            
            # Process each resume in the data directory
            for industry_dir in self.raw_resumes_dir.glob("*"):
                if not industry_dir.is_dir():
                    continue
                    
                logger.info(f"Processing resumes in {industry_dir.name}...")
                for pdf_file in industry_dir.glob("*.pdf"):
                    try:
                        # Extract text from PDF
                        text = self.extract_text_from_pdf(pdf_file)
                        if not text:
                            continue
                            
                        # Extract information
                        contact_info = self.extract_contact_info(text)
                        education = self.extract_education(text)
                        experience = self.extract_experience(text)
                        skills = self.extract_skills(text)
                        
                        # Create candidate profile
                        candidate = CandidateProfile(
                            candidate_id=pdf_file.stem,
                            name="",  # Would need more sophisticated name extraction
                            email=contact_info['email'],
                            phone=contact_info['phone'],
                            location=contact_info['location'],
                            summary=text[:500],  # Use first 500 characters as summary
                            skills=skills,
                            experience=[Experience(**exp) for exp in experience],
                            education=[Education(**edu) for edu in education],
                            languages=[],  # Would need language detection
                            certifications=[],  # Would need certification extraction
                            linkedin_url=contact_info['linkedin_url'],
                            portfolio_url=contact_info['portfolio_url']
                        )
                        processed_candidates.append(candidate)
                        
                    except Exception as e:
                        logger.warning(f"Error processing resume {pdf_file}: {str(e)}")
                        continue
            
            # Save processed data
            self.save_json_data(
                [candidate.dict() for candidate in processed_candidates],
                self.processed_candidates_file
            )
            return processed_candidates
            
        except Exception as e:
            logger.error(f"Error processing candidates: {str(e)}")
            raise DataIngestionError(f"Failed to process candidates: {str(e)}") 