import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from data_models import (
    JobPosting, CandidateProfile, WorkExperience, Education,
    ContactInfo, CertificationOrAward, validate_json
)
import PyPDF2
from io import StringIO
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from datetime import datetime
import email_validator
from pydantic import BaseModel

DATA_ROOT = Path(__file__).parent.parent / "data"

class JobPosting(BaseModel):
    job_id: str
    title: str
    description: str
    requirements: str
    company: str
    location: str
    benefits: List[str] = []
    industries: List[str] = []
    skills: List[str] = []
    salary: Optional[Dict[str, Any]] = None
    company_info: Optional[Dict[str, Any]] = None
    work_type: Optional[str] = None
    experience_level: Optional[str] = None
    remote_allowed: Optional[bool] = None

def load_companies() -> Dict[str, Dict]:
    """Load and process company data"""
    companies = {}
    
    # Load core company data
    with open(DATA_ROOT / "job-postings/companies/companies.csv") as f:
        for row in csv.DictReader(f):
            companies[row['company_id']] = {
                'name': row['name'],
                'industries': [],
                'specialities': [],
                'employee_count': None,
                'linkedin_url': row.get('url'),
                'website': row.get('url')
            }
    
    # Add industries
    with open(DATA_ROOT / "job-postings/companies/company_industries.csv") as f:
        for row in csv.DictReader(f):
            if row['company_id'] in companies:
                companies[row['company_id']]['industries'].append(row['industry'])
    
    # Add specialities
    with open(DATA_ROOT / "job-postings/companies/company_specialities.csv") as f:
        for row in csv.DictReader(f):
            if row['company_id'] in companies:
                companies[row['company_id']]['specialities'].append(row['speciality'])
    
    # Add employee counts
    with open(DATA_ROOT / "job-postings/companies/employee_counts.csv") as f:
        for row in csv.DictReader(f):
            if row['company_id'] in companies:
                companies[row['company_id']]['employee_count'] = {
                    'min': row.get('employee_count_min'),
                    'max': row.get('employee_count_max')
                }
    
    return companies

def load_mappings() -> Dict[str, Dict]:
    """Load mapping data for industries and skills"""
    mappings = {
        'industries': {},
        'skills': {}
    }
    
    # Load industry mappings
    with open(DATA_ROOT / "job-postings/mappings/industries.csv") as f:
        for row in csv.DictReader(f):
            mappings['industries'][row['industry_id']] = row['industry_name']
    
    # Load skill mappings
    with open(DATA_ROOT / "job-postings/mappings/skills.csv") as f:
        for row in csv.DictReader(f):
            mappings['skills'][row['skill_abr']] = row['skill_name']
    
    return mappings

def str_to_bool(value: str) -> Optional[bool]:
    """Convert string to boolean or None"""
    if not value or value.lower() in ('', 'n/a', 'none', 'null'):
        return None
    return value.lower() in ('true', '1', 't', 'y', 'yes')

def load_job_postings() -> List[JobPosting]:
    """Process job posting data from CSV"""
    processed = []
    
    # Load companies and mappings first
    companies = load_companies()
    mappings = load_mappings()
    
    # Load job benefits
    benefits = {}
    with open(DATA_ROOT / "job-postings/jobs/benefits.csv") as f:
        for row in csv.DictReader(f):
            if row['job_id'] not in benefits:
                benefits[row['job_id']] = []
            benefits[row['job_id']].append(row['type'])
    
    # Load job industries
    job_industries = {}
    with open(DATA_ROOT / "job-postings/jobs/job_industries.csv") as f:
        for row in csv.DictReader(f):
            if row['job_id'] not in job_industries:
                job_industries[row['job_id']] = []
            industry_name = mappings['industries'].get(row['industry_id'])
            if industry_name:
                job_industries[row['job_id']].append(industry_name)
    
    # Load job skills
    job_skills = {}
    with open(DATA_ROOT / "job-postings/jobs/job_skills.csv") as f:
        for row in csv.DictReader(f):
            if row['job_id'] not in job_skills:
                job_skills[row['job_id']] = []
            skill_name = mappings['skills'].get(row['skill_abr'])
            if skill_name:
                job_skills[row['job_id']].append(skill_name)
    
    # Load salary information
    salaries = {}
    with open(DATA_ROOT / "job-postings/jobs/salaries.csv") as f:
        for row in csv.DictReader(f):
            salaries[row['job_id']] = {
                'min_salary': row.get('min_salary'),
                'max_salary': row.get('max_salary'),
                'currency': row.get('currency', 'USD'),
                'compensation_type': row.get('compensation_type')
            }
    
    # Load core job data and combine all information
    with open(DATA_ROOT / "job-postings/postings.csv") as f:
        for job in csv.DictReader(f):
            try:
                job_id = job['job_id']
                company_id = job.get('company_id')
                company_info = companies.get(company_id, {})
                
                # Get job-specific skills
                skills = job_skills.get(job_id, [])
                if job.get('skills_desc'):
                    # Add any additional skills from description
                    skills.extend([s.strip() for s in job['skills_desc'].split(',') if s.strip()])
                skills = list(set(skills))  # Remove duplicates
                
                # Build company info
                company_details = None
                if company_info:
                    company_details = {
                        'industries': company_info.get('industries', []),
                        'specialities': company_info.get('specialities', []),
                        'employee_count': company_info.get('employee_count'),
                        'linkedin_url': company_info.get('linkedin_url'),
                        'website': company_info.get('website')
                    }
                
                # Create job posting
                processed.append(JobPosting(
                    job_id=job_id,
                    title=job['title'],
                    description=job['description'],
                    requirements=job.get('requirements', ''),
                    company=company_info.get('name', job.get('company_name', '')),
                    location=job['location'],
                    benefits=benefits.get(job_id, []),
                    industries=job_industries.get(job_id, []),
                    skills=skills,
                    salary=salaries.get(job_id),
                    company_info=company_details,
                    work_type=job.get('formatted_work_type'),
                    experience_level=job.get('formatted_experience_level'),
                    remote_allowed=str_to_bool(job.get('remote_allowed'))
                ))
            except Exception as e:
                print(f"Error processing job {job.get('job_id', 'unknown')}: {str(e)}")
                continue
    
    return processed

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using pdfminer for better text extraction"""
    try:
        return extract_text(str(pdf_path), laparams=LAParams())
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def find_pdf_by_id(resume_id: str) -> Optional[Path]:
    """Find PDF file by resume ID across all industry folders"""
    pdf_root = DATA_ROOT / "resume/data/data"
    
    # Search in all industry folders
    for industry_dir in pdf_root.glob("*"):
        if not industry_dir.is_dir():
            continue
            
        pdf_path = industry_dir / f"{resume_id}.pdf"
        if pdf_path.exists():
            return pdf_path, industry_dir.name
    
    return None, None

def extract_contact_info(text: str) -> ContactInfo:
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
    
    return ContactInfo(
        email=emails[0] if emails else None,
        phone=phones[0] if phones else None,
        location=locations[0] if locations else None,
        linkedin=f"linkedin.com/in/{linkedin[0]}" if linkedin else None,
        website=websites[0] if websites and 'linkedin' not in websites[0] else None
    )

def extract_date(text: str) -> Optional[str]:
    """Extract and normalize date from text"""
    # Various date patterns
    patterns = [
        r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+\d{4})',
        r'(\d{1,2}/\d{4})',
        r'(\d{2}/\d{2}/\d{2,4})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                date_str = match.group(1)
                # Try to parse and normalize the date
                # For now, return as string, can be enhanced to return datetime
                return date_str
            except:
                continue
    return None

def extract_work_experience(text: str) -> List[WorkExperience]:
    """Extract detailed work experience entries"""
    experiences = []
    
    # Look for experience section
    exp_sections = re.split(r'(?i)(?:EXPERIENCE|WORK HISTORY|EMPLOYMENT|PROFESSIONAL BACKGROUND)[:\s]*', text)
    if len(exp_sections) < 2:
        return experiences
    
    exp_text = exp_sections[1]
    # Stop at next major section
    next_section = re.search(r'\n\n+(?:[A-Z][A-Z\s]+:?|EDUCATION|SKILLS|CERTIFICATIONS)', exp_text)
    if next_section:
        exp_text = exp_text[:next_section.start()]
    
    # First try to find all job entries by looking for date patterns
    date_patterns = [
        # Standard date range pattern
        r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|(?:\d{1,2}/\d{4}))\s*(?:to|–|-)\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|(?:\d{1,2}/\d{4})|Current|Present|Now)',
        # Just years pattern
        r'\b\d{4}\s*(?:to|–|-)\s*(?:\d{4}|Current|Present|Now)\b',
        # Single date with "to Current/Present"
        r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|(?:\d{1,2}/\d{4})|(?:\d{4}))\s*(?:to|–|-)\s*(?:Current|Present|Now)',
        # Date at end of line
        r'(?:^|\n).*?(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|(?:\d{1,2}/\d{4}))\s*$'
    ]
    
    # Find all date ranges
    date_matches = []
    for pattern in date_patterns:
        matches = list(re.finditer(pattern, exp_text))
        if matches:
            date_matches.extend(matches)
            break  # Use first successful pattern
    
    if not date_matches:
        return experiences
    
    # Sort matches by position
    date_matches.sort(key=lambda x: x.start())
    
    # Process each section between dates
    for i in range(len(date_matches)):
        date_range = date_matches[i].group(0)
        
        # Parse dates
        if 'to' in date_range or '–' in date_range or '-' in date_range:
            dates = re.split(r'\s*(?:to|–|-)\s*', date_range)
            start_date = dates[0].strip()
            end_date = dates[1].strip() if len(dates) > 1 else "Present"
        else:
            # Single date format
            start_date = date_range.strip()
            end_date = "Present"
        
        # Get text before this date (up to previous date or section start)
        if i > 0:
            prev_end = date_matches[i-1].end()
            header_text = exp_text[prev_end:date_matches[i].start()]
        else:
            header_text = exp_text[:date_matches[i].start()]
        
        # Get text after this date (up to next date or section end)
        if i < len(date_matches) - 1:
            desc_text = exp_text[date_matches[i].end():date_matches[i+1].start()]
        else:
            desc_text = exp_text[date_matches[i].end():]
            # Stop at next section if found
            next_section = re.search(r'\n\n+(?:[A-Z][A-Z\s]+:?|EDUCATION|SKILLS|CERTIFICATIONS)', desc_text)
            if next_section:
                desc_text = desc_text[:next_section.start()]
        
        # Combine header and description
        entry_text = header_text + "\n" + desc_text
        
        # Split into lines and process
        lines = [line.strip() for line in entry_text.split('\n') if line.strip()]
        if not lines:
            continue
        
        # First non-empty line that's not a bullet point is likely the title/company
        title_line = None
        for line in lines:
            if line and not line.startswith(('•', '-', '•', '∙', '○', '●')):
                # Skip if line contains a date
                if not any(re.search(p, line) for p in date_patterns):
                    title_line = line
                    break
        
        if not title_line:
            continue
        
        # Try to split title and company
        title = None
        company = None
        
        # Try different patterns to split title and company
        patterns = [
            r'^(.*?)\s+(?:at|@|,|\|)\s+(.+)$',  # Title at Company
            r'^(.*?)\s{2,}(.+)$',  # Title  Company (multiple spaces)
            r'^([^,]+),\s*(.+)$',  # Title, Company
            r'^(.+?)(?:\s+-\s+|\s*\|\s*)(.+)$'  # Title - Company or Title | Company
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title_line)
            if match:
                title = match.group(1).strip()
                company = match.group(2).strip()
                break
        
        if not title:
            # If no pattern matches, use the whole line as title
            title = title_line
            company = "Unknown Company"
        
        # Clean up title and company
        title = re.sub(r'\s*(?:,|\|).*$', '', title)  # Remove anything after comma or pipe
        company = re.sub(r'\s*(?:,|\|).*$', '', company)  # Remove anything after comma or pipe
        
        # Extract location if present
        location = None
        location_pattern = r'(?:^|\n)([A-Z][a-zA-Z\s.-]+,\s*[A-Z]{2,})'
        location_match = re.search(location_pattern, entry_text)
        if location_match:
            location = location_match.group(1)
            # Remove location from company if it's there
            company = company.replace(location, '').strip()
        
        # Clean up company name
        company = re.sub(r'\s*(?:Company Name|Ltd\.?|Inc\.?|Corp\.?|Corporation)\s*', '', company).strip()
        company = re.sub(r'\s+', ' ', company)  # Normalize whitespace
        company = re.sub(r'\s*\u00ef\u00bc\u200b\s*', '', company)  # Remove special characters
        
        # Remaining lines are description bullets
        description = []
        current_line = ""
        
        for line in lines[1:]:
            line = line.strip('•').strip('-').strip('•').strip('∙').strip('○').strip('●').strip()
            
            # Skip if line contains noise
            if not line or any(line.startswith(x) for x in ['Location:', 'Address:', 'Based in:', 'Education:', 'Skills:']):
                continue
                
            # Skip if line looks like a section header or contains noise
            if re.match(r'^[A-Z\s]+:?$', line):  # Skip all-caps section headers
                continue
                
            if any(x in line.lower() for x in ['education:', 'skills:', 'certification:', 'summary:', 'experience:']):
                continue
            
            # Skip if line contains a date
            if any(re.search(p, line) for p in date_patterns):
                continue
            
            # Combine split lines
            if len(line) < 50 and not line.endswith('.'):
                current_line += " " + line
            else:
                if current_line:
                    description.append(current_line.strip())
                    current_line = ""
                if len(line) > 10:  # Ignore very short lines
                    description.append(line)
        
        # Add last line if any
        if current_line:
            description.append(current_line.strip())
        
        # Clean up description
        description = [d for d in description if len(d) > 10 and not any(x in d.lower() for x in [
            'education', 'skills', 'certification', 'summary', 'experience',
            'university', 'college', 'degree', 'gpa', 'major', 'minor'
        ])]
        
        # Only add if we have a reasonable title and it's not a section header
        if (len(title) > 2 and not title.isupper() and 
            not any(x in title.lower() for x in ['education', 'skills', 'summary', 'experience', 'background'])):
            experiences.append(WorkExperience(
                title=title,
                company=company,
                location=location,
                start_date=start_date,
                end_date=end_date,
                description=description
            ))
    
    return experiences

def extract_education(text: str) -> List[Education]:
    """Extract detailed education information"""
    education_list = []
    
    # Look for education section
    edu_sections = re.split(r'(?i)(?:EDUCATION|ACADEMIC)[:\s]*', text)
    if len(edu_sections) < 2:
        return education_list
    
    edu_text = edu_sections[1]
    # Stop at next major section
    next_section = re.search(r'\n\n+(?:[A-Z][A-Z\s]+:?|EXPERIENCE|SKILLS|CERTIFICATIONS)', edu_text)
    if next_section:
        edu_text = edu_text[:next_section.start()]
    
    # Split into entries
    entries = [e.strip() for e in re.split(r'\n(?=[A-Z])', edu_text) if e.strip()]
    
    # Common degree terms and noise words to filter out
    degree_terms = [
        'Bachelor', 'Master', 'PhD', 'B.S.', 'M.S.', 'B.A.', 'M.A.', 'M.B.A.', 'Ph.D.',
        'Associate', 'Diploma', 'Certificate', 'Degree'
    ]
    noise_words = [
        'skills', 'experience', 'summary', 'history', 'background',
        'certification', 'award', 'resume', 'cv', 'profile', 'objective',
        'software', 'tools', 'technologies', 'programming', 'languages'
    ]
    
    for entry in entries:
        # Skip if too short or contains noise words
        if len(entry) < 5 or any(word in entry.lower() for word in noise_words):
            continue
            
        # Try to identify degree and institution
        degree = None
        institution = None
        
        # First try to find a known degree term
        for term in degree_terms:
            match = re.search(fr'{term}\s*(?:of|in)?\s*[^,\n]+', entry, re.IGNORECASE)
            if match:
                degree = match.group(0)
                # Rest might be institution
                rest = entry.replace(degree, '').strip(' ,-')
                if rest:
                    institution = rest.split('\n')[0]
                break
        
        # If no degree found, try other patterns
        if not degree:
            # Try "Institution - Degree" pattern
            parts = re.split(r'\s*[-,]\s*', entry.split('\n')[0])
            if len(parts) >= 2:
                institution = parts[0]
                degree = parts[1]
            else:
                # Just take the first line as degree
                degree = entry.split('\n')[0]
        
        # Extract graduation date
        grad_date = None
        date_match = re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{1,2}/\d{4}|\d{4}', entry)
        if date_match:
            grad_date = date_match.group(0)
            # Clean up degree and institution from the date
            if degree:
                degree = degree.replace(grad_date, '').strip()
            if institution:
                institution = institution.replace(grad_date, '').strip()
        
        # Extract GPA if present
        gpa = None
        gpa_match = re.search(r'GPA:?\s*(\d+\.\d+)', entry, re.IGNORECASE)
        if gpa_match:
            try:
                gpa = float(gpa_match.group(1))
            except:
                pass
        
        # Extract major/minor
        major = None
        minor = None
        
        major_match = re.search(r'(?:Major|Concentration|Field)(?:\sof\sStudy)?:\s*([^,\n]+)', entry, re.IGNORECASE)
        if major_match:
            major = major_match.group(1).strip()
        elif 'in' in (degree or '').lower():
            major_match = re.search(r'in\s+([^,\n]+)', degree, re.IGNORECASE)
            if major_match:
                major = major_match.group(1).strip()
                degree = re.sub(r'\s+in\s+[^,\n]+', '', degree, flags=re.IGNORECASE)
        
        minor_match = re.search(r'Minor:\s*([^,\n]+)', entry, re.IGNORECASE)
        if minor_match:
            minor = minor_match.group(1).strip()
        
        # Extract location if present
        location = None
        location_match = re.search(r'(?:^|\n)([A-Z][a-zA-Z\s.-]+,\s*[A-Z]{2,})', entry)
        if location_match:
            location = location_match.group(1)
        
        # Clean up and validate
        if degree and len(degree.strip()) > 2:  # Avoid single letters
            # Clean up common noise
            degree = re.sub(r'\s*(?:,|\|).*$', '', degree)
            if institution:
                institution = re.sub(r'\s*(?:,|\|).*$', '', institution)
            
            # Skip if degree or institution contains noise words or looks like a skill
            if not any(word in degree.lower() for word in noise_words) and not any(word in (institution or '').lower() for word in noise_words):
                if not re.match(r'^[A-Za-z\s.]+$', degree):  # Skip if degree is just a word (likely a skill)
                    education_list.append(Education(
                        degree=degree.strip(),
                        institution=institution.strip() if institution else "Unknown Institution",
                        location=location,
                        graduation_date=grad_date,
                        gpa=gpa,
                        major=major,
                        minor=minor
                    ))
    
    return education_list

def extract_certifications_and_awards(text: str) -> Tuple[List[CertificationOrAward], List[CertificationOrAward]]:
    """Extract certifications and awards"""
    certifications = []
    awards = []
    
    # Look for certifications section
    cert_sections = re.split(r'(?i)(?:CERTIFICATIONS?|LICENSES?)[:\s]*', text)
    if len(cert_sections) > 1:
        cert_text = cert_sections[1]
        cert_splits = re.split(r'\n(?=[A-Z])', cert_text)
        
        for cert in cert_splits:
            if not cert.strip():
                continue
                
            lines = cert.strip().split('\n')
            if not lines:
                continue
            
            name = lines[0].strip()
            date = extract_date(cert)
            
            # Try to extract issuer
            issuer_pattern = r'(?:issued by|from|through)\s+([^\n,]+)'
            issuer_match = re.search(issuer_pattern, cert, re.IGNORECASE)
            issuer = issuer_match.group(1).strip() if issuer_match else None
            
            certifications.append(CertificationOrAward(
                name=name,
                issuer=issuer,
                date=date
            ))
    
    # Look for awards section
    award_sections = re.split(r'(?i)(?:AWARDS?|HONORS?|ACHIEVEMENTS?)[:\s]*', text)
    if len(award_sections) > 1:
        award_text = award_sections[1]
        award_splits = re.split(r'\n(?=[A-Z])', award_text)
        
        for award in award_splits:
            if not award.strip():
                continue
                
            lines = award.strip().split('\n')
            if not lines:
                continue
            
            name = lines[0].strip()
            date = extract_date(award)
            description = ' '.join(lines[1:]).strip() if len(lines) > 1 else None
            
            awards.append(CertificationOrAward(
                name=name,
                date=date,
                description=description
            ))
    
    return certifications, awards

def extract_languages(text: str) -> List[str]:
    """Extract language skills"""
    languages = []
    
    # Look for language section
    lang_sections = re.split(r'(?i)(?:LANGUAGES?|LANGUAGE SKILLS?)[:\s]*', text)
    if len(lang_sections) > 1:
        lang_text = lang_sections[1].split('\n')[0]  # Take first line after header
        
        # Split on common separators and clean up
        langs = re.split(r'[,;]', lang_text)
        languages = [lang.strip() for lang in langs if lang.strip()]
    
    return languages

def extract_summary(text: str) -> Optional[str]:
    """Extract professional summary/objective"""
    summary_patterns = [
        r'(?i)(?:SUMMARY|PROFESSIONAL SUMMARY|OBJECTIVE)[:\s]*([^\n]+(?:\n(?![A-Z][A-Z\s]+:)[^\n]+)*)',
        r'(?i)(?:PROFILE|ABOUT)[:\s]*([^\n]+(?:\n(?![A-Z][A-Z\s]+:)[^\n]+)*)'
    ]
    
    for pattern in summary_patterns:
        match = re.search(pattern, text)
        if match:
            summary = match.group(1).strip()
            return summary
    
    return None

def extract_skills(text: str) -> List[str]:
    """Extract skills from text, filtering out noise"""
    skills = set()
    
    # Look for skills section
    skill_sections = re.split(r'(?i)(?:SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES|EXPERTISE|QUALIFICATIONS)[:\s]*', text)
    if len(skill_sections) > 1:
        skill_text = skill_sections[1]
        # Stop at next major section
        next_section = re.search(r'\n\n+(?:[A-Z][A-Z\s]+:?|EXPERIENCE|EDUCATION|CERTIFICATIONS)', skill_text)
        if next_section:
            skill_text = skill_text[:next_section.start()]
        
        # Split by common separators
        for skill in re.split(r'[,•\n•∙○●]', skill_text):
            skill = skill.strip()
            if skill and len(skill) < 50:  # Reasonable skill length
                # Filter out noise
                if not any(x in skill.lower() for x in [
                    'experience', 'education', 'summary', 'history', 'background',
                    'certification', 'award', 'degree', 'university', 'college',
                    'resume', 'cv', 'curriculum vitae', 'profile', 'objective'
                ]):
                    if not re.search(r'\d{4}', skill):  # Skip if contains year
                        if not re.match(r'^[A-Z\s]+$', skill):  # Skip all-caps section headers
                            if len(skill.split()) <= 4:  # Most skills are 1-4 words
                                skills.add(skill)
    
    return list(skills)

def extract_resume_sections(text: str) -> Tuple[str, List[str], ContactInfo, Optional[str], List[WorkExperience], List[Education], List[CertificationOrAward], List[CertificationOrAward], List[str]]:
    """Extract all structured information from resume text"""
    # Clean up the text
    text = text.replace('\r', '\n')
    text = re.sub(r'\n+', '\n', text)
    
    # Try to extract name from the beginning
    name = "Unknown"
    first_line = text.split('\n')[0].strip()
    if first_line and len(first_line) < 50:  # Reasonable name length
        name = first_line
    
    # Extract contact information
    contact_info = extract_contact_info(text)
    
    # Extract summary
    summary = extract_summary(text)
    
    # Extract work experience
    experience = extract_work_experience(text)
    
    # Extract education
    education = extract_education(text)
    
    # Extract skills using improved function
    skills = extract_skills(text)
    
    # Extract certifications and awards
    certifications, awards = extract_certifications_and_awards(text)
    
    # Extract languages
    languages = extract_languages(text)
    
    return name, skills, contact_info, summary, experience, education, certifications, awards, languages

def load_candidates() -> List[CandidateProfile]:
    """Process resume data from both CSV and PDF files"""
    candidates = []
    resume_file = DATA_ROOT / "resume/Resume/Resume.csv"
    
    if not resume_file.exists():
        print(f"Warning: Resume file {resume_file} does not exist")
        return candidates
    
    with open(resume_file) as f:
        reader = csv.DictReader(f)
        total_rows = sum(1 for row in csv.DictReader(open(resume_file)))
        
        # Reset file pointer
        f.seek(0)
        next(reader)  # Skip header
        
        for idx, row in enumerate(reader):
            try:
                resume_id = row.get('ID')
                if not resume_id:
                    continue
                
                print(f"\rProcessing resume {idx+1}/{total_rows}...", end="", flush=True)
                
                # Find corresponding PDF file
                pdf_path, industry = find_pdf_by_id(resume_id)
                
                # Extract text from both sources
                csv_text = row.get('Resume_str', '')
                pdf_text = extract_text_from_pdf(pdf_path) if pdf_path else ""
                
                # Combine texts, giving preference to PDF content if available
                combined_text = pdf_text if pdf_text.strip() else csv_text
                
                # Extract all structured information
                name, skills, contact_info, summary, experience, education, certifications, awards, languages = extract_resume_sections(combined_text)
                
                # Create candidate profile
                candidate = validate_json({
                    "candidate_id": resume_id,
                    "name": name,
                    "contact_info": contact_info,
                    "summary": summary,
                    "experience": experience,
                    "education": education,
                    "skills": list(set(skills))[:20],  # Limit to top 20 unique skills
                    "certifications": certifications,
                    "awards": awards,
                    "languages": languages,
                    "industry": industry or "Unknown",
                    "raw_text": combined_text[:1000]  # Store first 1000 chars for context
                }, CandidateProfile)
                
                candidates.append(candidate)
                
            except Exception as e:
                print(f"\nError processing candidate {row.get('ID', 'unknown')}: {str(e)}")
                continue
    
    print(f"\nCompleted processing {len(candidates)} resumes")
    return candidates

def save_structured_data(data: List, output_path: str):
    """Save structured data to JSON for pipeline consumption"""
    output_file = DATA_ROOT / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump([item.model_dump() for item in data], f, indent=2)

if __name__ == "__main__":
    # Process and save job data
    print("Processing job postings...")
    jobs = load_job_postings()
    save_structured_data(jobs, "processed/structured_jobs.json")
    print(f"Processed {len(jobs)} job postings")
    
    # Process and save candidate data
    print("\nProcessing candidate resumes...")
    candidates = load_candidates()
    save_structured_data(candidates, "processed/structured_candidates.json")
    print(f"Processed {len(candidates)} candidate resumes") 