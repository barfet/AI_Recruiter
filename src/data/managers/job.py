import csv
from typing import Dict, List, Optional

from src.data.models.job import JobPosting
from src.data.managers.base import BaseManager
from src.core.config import settings
from src.core.logging import setup_logger

logger = setup_logger(__name__)


class JobManager(BaseManager):
    """Manager for job posting data"""

    def __init__(self):
        super().__init__(JobPosting)
        self.companies_dir = settings.JOB_POSTINGS_DIR / "companies"
        self.jobs_dir = settings.JOB_POSTINGS_DIR / "jobs"
        self.mappings_dir = settings.JOB_POSTINGS_DIR / "mappings"
        self.postings_file = settings.JOB_POSTINGS_DIR / "postings.csv"

    def str_to_float(self, value: str) -> Optional[float]:
        """Convert string to float or None"""
        if not value or value.lower() in ("", "n/a", "none", "null"):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def load_companies(self) -> Dict[str, Dict]:
        """Load and process company data"""
        companies = {}

        try:
            # Load core company data
            with open(self.companies_dir / "companies.csv") as f:
                for row in csv.DictReader(f):
                    companies[row["company_id"]] = {
                        "name": row["name"],
                        "industries": [],
                        "specialities": [],
                        "employee_count": None,
                        "linkedin_url": row.get("url"),
                        "website": row.get("url"),
                    }

            # Add industries
            with open(self.companies_dir / "company_industries.csv") as f:
                for row in csv.DictReader(f):
                    if row["company_id"] in companies:
                        companies[row["company_id"]]["industries"].append(
                            row["industry"]
                        )

            # Add specialities
            with open(self.companies_dir / "company_specialities.csv") as f:
                for row in csv.DictReader(f):
                    if row["company_id"] in companies:
                        companies[row["company_id"]]["specialities"].append(
                            row["speciality"]
                        )

            # Add employee counts
            with open(self.companies_dir / "employee_counts.csv") as f:
                for row in csv.DictReader(f):
                    if row["company_id"] in companies:
                        companies[row["company_id"]]["employee_count"] = {
                            "min": self.str_to_float(row.get("employee_count_min")),
                            "max": self.str_to_float(row.get("employee_count_max")),
                        }
        except Exception as e:
            logger.error(f"Error loading company data: {str(e)}")
            raise

        return companies

    def load_mappings(self) -> Dict[str, Dict]:
        """Load mapping data for industries and skills"""
        mappings = {"industries": {}, "skills": {}}

        try:
            # Load industry mappings
            with open(self.mappings_dir / "industries.csv") as f:
                for row in csv.DictReader(f):
                    mappings["industries"][row["industry_id"]] = row["industry_name"]

            # Load skill mappings
            with open(self.mappings_dir / "skills.csv") as f:
                for row in csv.DictReader(f):
                    mappings["skills"][row["skill_abr"]] = row["skill_name"]
        except Exception as e:
            logger.error(f"Error loading mappings: {str(e)}")
            raise

        return mappings

    def str_to_bool(self, value: str) -> Optional[bool]:
        """Convert string to boolean or None"""
        if not value or value.lower() in ("", "n/a", "none", "null"):
            return None
        return value.lower() in ("true", "1", "t", "y", "yes")

    def _load_job_benefits(self) -> Dict[str, List[str]]:
        """Load job benefits data"""
        benefits = {}
        with open(self.jobs_dir / "benefits.csv") as f:
            for row in csv.DictReader(f):
                if row["job_id"] not in benefits:
                    benefits[row["job_id"]] = []
                benefits[row["job_id"]].append(row["type"])
        return benefits

    def _load_job_industries(self, mappings: Dict) -> Dict[str, List[str]]:
        """Load job industries data"""
        job_industries = {}
        with open(self.jobs_dir / "job_industries.csv") as f:
            for row in csv.DictReader(f):
                if row["job_id"] not in job_industries:
                    job_industries[row["job_id"]] = []
                industry_name = mappings["industries"].get(row["industry_id"])
                if industry_name:
                    job_industries[row["job_id"]].append(industry_name)
        return job_industries

    def _load_job_skills(self, mappings: Dict) -> Dict[str, List[str]]:
        """Load job skills data"""
        job_skills = {}
        with open(self.jobs_dir / "job_skills.csv") as f:
            for row in csv.DictReader(f):
                if row["job_id"] not in job_skills:
                    job_skills[row["job_id"]] = []
                skill_name = mappings["skills"].get(row["skill_abr"])
                if skill_name:
                    job_skills[row["job_id"]].append(skill_name)
        return job_skills

    def _load_salaries(self) -> Dict[str, Dict]:
        """Load salary information"""
        salaries = {}
        with open(self.jobs_dir / "salaries.csv") as f:
            for row in csv.DictReader(f):
                salaries[row["job_id"]] = {
                    "min_salary": self.str_to_float(row.get("min_salary")),
                    "max_salary": self.str_to_float(row.get("max_salary")),
                    "currency": row.get("currency", "USD"),
                    "compensation_type": row.get("compensation_type"),
                }
        return salaries

    def _create_job_posting(
        self,
        *,  # Force keyword arguments
        job: Dict,
        companies: Dict,
        benefits: Dict[str, List[str]],
        job_industries: Dict[str, List[str]],
        job_skills: Dict[str, List[str]],
        salaries: Dict[str, Dict]
    ) -> JobPosting:
        """Create a job posting from raw data"""
        job_id = job["job_id"]
        company_id = job.get("company_id")
        company_info = companies.get(company_id, {})

        # Get job-specific skills
        skills = job_skills.get(job_id, [])
        if job.get("skills_desc"):
            # Add any additional skills from description
            skills.extend([s.strip() for s in job["skills_desc"].split(",") if s.strip()])
        skills = list(set(skills))  # Remove duplicates

        # Build company info
        company_details = None
        if company_info:
            company_details = {
                "industries": company_info.get("industries", []),
                "specialities": company_info.get("specialities", []),
                "employee_count": company_info.get("employee_count"),
                "linkedin_url": company_info.get("linkedin_url"),
                "website": company_info.get("website"),
            }

        # Create job posting
        job_data = {
            "job_id": job_id,
            "title": job["title"],
            "description": job["description"],
            "requirements": job.get("requirements", ""),
            "company": company_info.get("name", job.get("company_name", "")),
            "location": job["location"],
            "benefits": benefits.get(job_id, []),
            "industries": job_industries.get(job_id, []),
            "skills": skills,
            "salary": salaries.get(job_id),
            "company_info": company_details,
            "work_type": job.get("formatted_work_type"),
            "experience_level": job.get("formatted_experience_level"),
            "remote_allowed": self.str_to_bool(job.get("remote_allowed")),
        }

        return self.validate_data(job_data)

    def process_jobs(self) -> List[JobPosting]:
        """Process job posting data from CSV"""
        processed = []

        try:
            # Load supporting data
            companies = self.load_companies()
            mappings = self.load_mappings()
            benefits = self._load_job_benefits()
            job_industries = self._load_job_industries(mappings)
            job_skills = self._load_job_skills(mappings)
            salaries = self._load_salaries()

            # Process core job data
            with open(self.postings_file) as f:
                for job in csv.DictReader(f):
                    try:
                        job_posting = self._create_job_posting(
                            job=job,
                            companies=companies,
                            benefits=benefits,
                            job_industries=job_industries,
                            job_skills=job_skills,
                            salaries=salaries,
                        )
                        processed.append(job_posting)
                    except Exception as e:
                        logger.warning(
                            f"Error processing job {job.get('job_id', 'unknown')}: {str(e)}"
                        )
                        continue

            # Save processed data
            self.save_processed_data(processed, "structured_jobs.json")

        except Exception as e:
            logger.error(f"Error processing jobs: {str(e)}")
            raise

        return processed
