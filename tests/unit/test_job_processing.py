"""Unit tests for job data ingestion and processing."""

import pytest
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.data.managers.job import JobManager
from src.core.config import get_config
from tests.unit.base import BaseUnitTest

class TestJobProcessing(BaseUnitTest):
    """Test suite for job data processing."""
    
    def setup_method(self) -> None:
        """Set up test environment."""
        super().setup_method()
        self.manager = JobManager()
        self.test_data = self._create_test_data()
        
    def _create_test_data(self) -> Dict[str, pd.DataFrame]:
        """Create test job data."""
        postings = pd.DataFrame({
            "job_id": ["job1", "job2"],
            "title": ["Senior Python Developer", "Data Scientist"],
            "description": [
                "Looking for Python expert with web dev experience",
                "ML engineer position with focus on NLP"
            ],
            "location": ["New York", "San Francisco"],
            "company_id": ["comp1", "comp2"]
        })
        
        benefits = pd.DataFrame({
            "job_id": ["job1", "job1", "job2"],
            "benefit": ["Health Insurance", "401k", "Remote Work"]
        })
        
        company_industries = pd.DataFrame({
            "company_id": ["comp1", "comp2"],
            "industry": ["Technology", "AI/ML"]
        })
        
        skills = pd.DataFrame({
            "job_id": ["job1", "job1", "job2", "job2"],
            "skill": ["Python", "Django", "Python", "TensorFlow"]
        })
        
        return {
            "postings": postings,
            "benefits": benefits,
            "company_industries": company_industries,
            "skills": skills
        }
    
    def _create_csv_files(self, tmp_path: Path) -> Dict[str, Path]:
        """Create test CSV files."""
        files = {}
        for name, df in self.test_data.items():
            file_path = tmp_path / f"{name}.csv"
            df.to_csv(file_path, index=False)
            files[name] = file_path
        return files
    
    async def test_job_manager_ingestion(self, tmp_path: Path) -> None:
        """Test CSV-based job ingestion."""
        test_files = self._create_csv_files(tmp_path)
        
        # Process job data
        result = await self.manager.process_jobs(
            postings_file=test_files["postings"],
            benefits_file=test_files["benefits"],
            industries_file=test_files["company_industries"],
            skills_file=test_files["skills"],
            force=True
        )
        
        assert len(result) == 2  # Two job postings
        
        # Verify first job
        job1 = next(job for job in result if job["id"] == "job1")
        assert job1["title"] == "Senior Python Developer"
        assert job1["location"] == "New York"
        assert set(job1["benefits"]) == {"Health Insurance", "401k"}
        assert set(job1["skills"]) == {"Python", "Django"}
        assert job1["industry"] == "Technology"
        
        # Verify second job
        job2 = next(job for job in result if job["id"] == "job2")
        assert job2["title"] == "Data Scientist"
        assert "ML" in job2["description"]
        assert "Remote Work" in job2["benefits"]
        assert "TensorFlow" in job2["skills"]
        
        # Verify all required fields are present
        required_fields = {"id", "title", "description", "location", "skills", "benefits", "industry"}
        assert all(field in job1 for field in required_fields)
        assert all(field in job2 for field in required_fields)
    
    async def test_job_edge_cases(self, tmp_path: Path) -> None:
        """Test edge cases in job processing."""
        # Test missing files
        with pytest.raises(FileNotFoundError):
            await self.manager.process_jobs(
                postings_file=tmp_path / "nonexistent.csv"
            )
        
        # Test empty dataframes
        empty_files = {}
        for name in self.test_data.keys():
            file_path = tmp_path / f"empty_{name}.csv"
            pd.DataFrame(columns=self.test_data[name].columns).to_csv(file_path, index=False)
            empty_files[name] = file_path
        
        result = await self.manager.process_jobs(
            postings_file=empty_files["postings"],
            benefits_file=empty_files["benefits"],
            industries_file=empty_files["company_industries"],
            skills_file=empty_files["skills"]
        )
        assert len(result) == 0
        
        # Test missing columns
        invalid_df = pd.DataFrame({"wrong_column": ["data"]})
        invalid_file = tmp_path / "invalid.csv"
        invalid_df.to_csv(invalid_file, index=False)
        
        with pytest.raises(Exception):  # Should raise on missing required columns
            await self.manager.process_jobs(postings_file=invalid_file)
    
    async def test_job_data_cleaning(self, tmp_path: Path) -> None:
        """Test job data cleaning and normalization."""
        test_files = self._create_csv_files(tmp_path)
        
        # Modify test data to include messy data
        messy_df = pd.read_csv(test_files["postings"])
        messy_df.loc[0, "location"] = "NEW YORK, NY"  # Test location normalization
        messy_df.loc[1, "title"] = " Data Scientist "  # Test whitespace cleaning
        
        messy_file = test_files["postings"].parent / "messy_postings.csv"
        messy_df.to_csv(messy_file, index=False)
        
        result = await self.manager.process_jobs(
            postings_file=messy_file,
            benefits_file=test_files["benefits"],
            industries_file=test_files["company_industries"],
            skills_file=test_files["skills"]
        )
        
        # Verify cleaning
        job1 = next(job for job in result if job["id"] == "job1")
        assert job1["location"] == "New York"  # Should be normalized
        
        job2 = next(job for job in result if job["id"] == "job2")
        assert job2["title"] == "Data Scientist"  # Should be stripped
    
    def teardown_method(self) -> None:
        """Clean up after test."""
        super().teardown_method() 