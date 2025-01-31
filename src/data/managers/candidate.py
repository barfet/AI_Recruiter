import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

from src.data.models.candidate import CandidateProfile
from src.data.managers.base import BaseManager
from src.core.config import settings
from src.core.logging import setup_logger

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
                all_texts=True
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
            'email': None,
            'phone': None,
            'location': None,
            'linkedin': None,
            'website': None
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group()
            
        # Phone pattern (various formats)
        phone_pattern = r'''(?:
            (?:\+\d{1,2}\s*)?                    # Optional country code
            \(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}  # (123) 456-7890 or 123-456-7890
        )'''
        phone_match = re.search(phone_pattern, text, re.VERBOSE)
        if phone_match:
            contact_info['phone'] = phone_match.group()
            
        # Location pattern (City, State or City, Country)
        location_pattern = r'(?:^|\n)(?:Address:|Location:|Based in:)?\s*([A-Z][a-zA-Z\s.-]+,\s*[A-Z]{2,})'
        location_match = re.search(location_pattern, text)
        if location_match:
            contact_info['location'] = location_match.group(1).strip()
            
        # LinkedIn pattern
        linkedin_pattern = r'(?:linkedin\.com/in/|LinkedIn:?\s*)([a-zA-Z0-9-/]+)'
        linkedin_match = re.search(linkedin_pattern, text.lower())
        if linkedin_match:
            contact_info['linkedin'] = f"linkedin.com/in/{linkedin_match.group(1)}"
            
        # Website pattern
        website_pattern = r'(?:Website:?\s*)?(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/\S*)?)'
        website_match = re.search(website_pattern, text.lower())
        if website_match and 'linkedin' not in website_match.group():
            contact_info['website'] = website_match.group()
            
        return contact_info
        
    def extract_education(self, text: str) -> List[Dict]:
        """Extract education information from resume text"""
        education = []
        
        # Common education keywords
        edu_keywords = r'(?:Bachelor|Master|PhD|B\.S\.|M\.S\.|M\.A\.|B\.A\.|Associate|Diploma)'
        degree_pattern = f'{edu_keywords}[\s\w]*(?:of|in|,)?\s*[\w\s,]*'
        
        # Date pattern
        date_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{4}'
        
        # Find education sections
        sections = re.split(r'\n{2,}', text)
        for section in sections:
            degree_match = re.search(degree_pattern, section, re.IGNORECASE)
            if degree_match:
                edu_entry = {
                    'degree': degree_match.group().strip(),
                    'institution': '',
                    'start_date': None,
                    'end_date': None
                }
                
                # Try to find institution name
                lines = section.split('\n')
                for line in lines:
                    if 'University' in line or 'College' in line or 'Institute' in line:
                        edu_entry['institution'] = line.strip()
                        break
                
                # Try to find dates
                dates = re.findall(date_pattern, section)
                if len(dates) >= 2:
                    edu_entry['start_date'] = dates[0]
                    edu_entry['end_date'] = dates[1]
                elif len(dates) == 1:
                    edu_entry['end_date'] = dates[0]
                
                education.append(edu_entry)
        
        return education
        
    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience from resume text"""
        experience = []
        
        # Date pattern
        date_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{4}'
        
        # Split into sections
        sections = re.split(r'\n{2,}', text)
        current_exp = None
        
        for section in sections:
            # Look for sections that might be job entries
            if re.search(date_pattern, section):
                if current_exp:
                    experience.append(current_exp)
                
                current_exp = {
                    'title': '',
                    'company': '',
                    'start_date': None,
                    'end_date': None,
                    'description': ''
                }
                
                # Extract dates
                dates = re.findall(date_pattern, section)
                if len(dates) >= 2:
                    current_exp['start_date'] = dates[0]
                    current_exp['end_date'] = dates[1]
                elif len(dates) == 1:
                    current_exp['start_date'] = dates[0]
                    if 'present' in section.lower():
                        current_exp['end_date'] = 'Present'
                
                # Extract title and company
                lines = section.split('\n')
                for line in lines:
                    if not current_exp['title'] and any(title in line.lower() for title in ['engineer', 'developer', 'manager', 'analyst', 'consultant']):
                        current_exp['title'] = line.strip()
                    elif not current_exp['company'] and len(line.strip()) > 0 and line.strip() != current_exp['title']:
                        current_exp['company'] = line.strip()
                        break
                
                # Get description
                desc_lines = []
                for line in lines[2:]:  # Skip title and company lines
                    if line.strip() and not re.search(date_pattern, line):
                        desc_lines.append(line.strip())
                current_exp['description'] = ' '.join(desc_lines)
        
        if current_exp:
            experience.append(current_exp)
        
        return experience
        
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        # Common skill keywords and patterns
        skill_patterns = [
            # Programming languages
            r'Python|Java|C\+\+|JavaScript|TypeScript|Ruby|PHP|Swift|Kotlin|Go|Rust',
            # Web technologies
            r'HTML5?|CSS3?|React|Angular|Vue\.js|Node\.js|Express\.js|Django|Flask|Spring|Laravel',
            # Databases
            r'SQL|MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle|SQLite',
            # Cloud platforms
            r'AWS|Azure|GCP|Docker|Kubernetes|Terraform|Jenkins|Git',
            # Machine Learning
            r'TensorFlow|PyTorch|Scikit-learn|Pandas|NumPy|OpenCV|NLP|Deep Learning',
            # Other technical skills
            r'REST|GraphQL|API|CI/CD|Agile|Scrum|Linux|Unix|DevOps'
        ]
        
        skills = set()
        combined_pattern = '|'.join(skill_patterns)
        
        # Find all matches
        matches = re.finditer(combined_pattern, text, re.IGNORECASE)
        for match in matches:
            skills.add(match.group())
            
        return sorted(list(skills))
        
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
                        candidate_data = {
                            'resume_id': resume_id,
                            'email': contact_info['email'],
                            'phone': contact_info['phone'],
                            'location': contact_info['location'],
                            'linkedin_url': contact_info['linkedin'],
                            'website': contact_info['website'],
                            'education': education,
                            'experience': experience,
                            'skills': skills,
                            'industry': industry_dir.name
                        }
                        
                        # Validate and add candidate
                        processed.append(self.validate_data(candidate_data))
                        
                    except Exception as e:
                        logger.warning(f"Error processing resume {pdf_file}: {str(e)}")
                        continue
                        
            # Save processed data
            self.save_processed_data(processed, "structured_candidates.json")
            
        except Exception as e:
            logger.error(f"Error processing candidates: {str(e)}")
            raise
            
        return processed 