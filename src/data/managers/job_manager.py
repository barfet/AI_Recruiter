from typing import Dict, List, Any, Optional
from pathlib import Path
import csv

from src.data.models.job import JobPosting
from src.data.managers.base import BaseDataManager
from src.core.logging import setup_logger
from src.core.exceptions import DataIngestionError

logger = setup_logger(__name__)

class JobManager(BaseDataManager):
    """Manager for job posting data"""
    
    def __init__(self):
        super().__init__(JobPosting)
        self.raw_jobs_file = self.raw_data_dir / "jobs/postings.csv"
        self.raw_companies_dir = self.raw_data_dir / "jobs/companies"
        self.raw_jobs_dir = self.raw_data_dir / "jobs/jobs"
        self.raw_mappings_dir = self.raw_data_dir / "jobs/mappings"
        self.processed_jobs_file = self.processed_data_dir / "structured_jobs.json"
        
    def load_companies(self) -> Dict[str, Dict]:
        """Load and process company data"""
        companies = {}
        
        # Load core company data
        with open(self.raw_companies_dir / "companies.csv") as f:
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
        with open(self.raw_companies_dir / "company_industries.csv") as f:
            for row in csv.DictReader(f):
                if row['company_id'] in companies:
                    companies[row['company_id']]['industries'].append(row['industry'])
        
        # Add specialities
        with open(self.raw_companies_dir / "company_specialities.csv") as f:
            for row in csv.DictReader(f):
                if row['company_id'] in companies:
                    companies[row['company_id']]['specialities'].append(row['speciality'])
        
        # Add employee counts
        with open(self.raw_companies_dir / "employee_counts.csv") as f:
            for row in csv.DictReader(f):
                if row['company_id'] in companies:
                    companies[row['company_id']]['employee_count'] = {
                        'min': row.get('employee_count_min'),
                        'max': row.get('employee_count_max')
                    }
        
        return companies
        
    def load_mappings(self) -> Dict[str, Dict]:
        """Load mapping data for industries and skills"""
        mappings = {
            'industries': {},
            'skills': {}
        }
        
        # Load industry mappings
        with open(self.raw_mappings_dir / "industries.csv") as f:
            for row in csv.DictReader(f):
                mappings['industries'][row['industry_id']] = row['industry_name']
        
        # Load skill mappings
        with open(self.raw_mappings_dir / "skills.csv") as f:
            for row in csv.DictReader(f):
                mappings['skills'][row['skill_abr']] = row['skill_name']
        
        return mappings
        
    def str_to_bool(self, value: str) -> bool:
        """Convert string to boolean or None"""
        if not value or value.lower() in ('', 'n/a', 'none', 'null'):
            return None
        return value.lower() in ('true', '1', 't', 'y', 'yes')
        
    def str_to_float(self, value: str) -> Optional[float]:
        """Convert string to float or None"""
        try:
            return float(value) if value else None
        except (ValueError, TypeError):
            return None
        
    def load_and_process_jobs(self) -> List[JobPosting]:
        """Load raw job data, process it, and return validated job postings"""
        try:
            # Load companies and mappings first
            companies = self.load_companies()
            mappings = self.load_mappings()
            
            # Load job benefits
            benefits = {}
            with open(self.raw_jobs_dir / "benefits.csv") as f:
                for row in csv.DictReader(f):
                    if row['job_id'] not in benefits:
                        benefits[row['job_id']] = []
                    benefits[row['job_id']].append(row['type'])
            
            # Load job industries
            job_industries = {}
            with open(self.raw_jobs_dir / "job_industries.csv") as f:
                for row in csv.DictReader(f):
                    if row['job_id'] not in job_industries:
                        job_industries[row['job_id']] = []
                    industry_name = mappings['industries'].get(row['industry_id'])
                    if industry_name:
                        job_industries[row['job_id']].append(industry_name)
            
            # Load job skills
            job_skills = {}
            with open(self.raw_jobs_dir / "job_skills.csv") as f:
                for row in csv.DictReader(f):
                    if row['job_id'] not in job_skills:
                        job_skills[row['job_id']] = []
                    skill_name = mappings['skills'].get(row['skill_abr'])
                    if skill_name:
                        job_skills[row['job_id']].append(skill_name)
            
            # Load salary information
            salaries = {}
            with open(self.raw_jobs_dir / "salaries.csv") as f:
                for row in csv.DictReader(f):
                    salaries[row['job_id']] = {
                        'min_salary': self.str_to_float(row.get('min_salary')),
                        'max_salary': self.str_to_float(row.get('max_salary')),
                        'currency': row.get('currency', 'USD'),
                        'compensation_type': row.get('compensation_type')
                    }
            
            # Process core job data
            processed_jobs = []
            with open(self.raw_jobs_file) as f:
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
                        processed_jobs.append(JobPosting(
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
                            remote_allowed=self.str_to_bool(job.get('remote_allowed'))
                        ))
                    except Exception as e:
                        logger.warning(f"Error processing job {job.get('job_id', 'unknown')}: {str(e)}")
                        continue
            
            # Save processed data
            self.save_json_data([job.dict() for job in processed_jobs], self.processed_jobs_file)
            return processed_jobs
            
        except Exception as e:
            logger.error(f"Error processing jobs: {str(e)}")
            raise DataIngestionError(f"Failed to process jobs: {str(e)}") 