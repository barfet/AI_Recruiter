"""Unit tests for candidate data parsing and cleaning."""

import pytest
from pathlib import Path
from typing import Dict, Any

from src.data.managers.candidate import CandidateManager
from src.core.config import get_config
from tests.unit.base import BaseUnitTest

class TestCandidateProcessing(BaseUnitTest):
    """Test suite for candidate data processing."""
    
    def setup_method(self) -> None:
        """Set up test environment."""
        super().setup_method()
        self.manager = CandidateManager()
        self.test_content = """
        John Doe
        Location: Chicago, IL
        
        Summary:
        Experienced software engineer with 5 years of Python development.
        
        Education:
        BSc in Computer Science, University of Illinois
        
        Skills:
        - Python, Django, Flask
        - AWS, Docker
        - SQL, PostgreSQL
        
        Experience:
        Software Engineer | TechCorp (2018-2023)
        - Developed Python web applications
        - Led team of 3 developers
        """
        
    def create_test_pdf(self, path: Path, content: str) -> Path:
        """Helper to create a test PDF file."""
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)
        
        pdf_path = path / f"test_{hash(content)}.pdf"
        pdf.output(str(pdf_path))
        return pdf_path
    
    async def test_candidate_parsing_and_cleaning(self, tmp_path: Path) -> None:
        """Test that a single candidate PDF is loaded, parsed, and cleaned correctly."""
        # Create test PDF
        pdf_file = self.create_test_pdf(tmp_path, self.test_content)
        
        # Process single PDF
        result = await self.manager.process_candidates(
            input_dir=pdf_file.parent,
            force=True
        )
        
        assert len(result) == 1
        candidate = result[0]
        
        # Verify basic fields
        assert candidate["location"] == "Chicago"
        assert "Python" in candidate["skills"]
        assert "AWS" in candidate["skills"]
        assert "Computer Science" in candidate["education"]
        
        # Verify experience parsing
        assert "5 years" in candidate["summary"]
        assert "TechCorp" in str(candidate["experience"])
        
        # Check for no unexpected entries
        assert all(isinstance(skill, str) for skill in candidate["skills"])
        assert all(isinstance(edu, str) for edu in candidate["education"])
    
    async def test_candidate_edge_cases(self, tmp_path: Path) -> None:
        """Test edge cases in candidate parsing."""
        # Test empty directory
        result = await self.manager.process_candidates(tmp_path)
        assert len(result) == 0
        
        # Test invalid PDF
        invalid_pdf = tmp_path / "invalid.pdf"
        invalid_pdf.write_bytes(b"Not a PDF file")
        
        with pytest.raises(Exception):
            await self.manager.process_candidates(invalid_pdf.parent)
        
        # Test missing required fields
        minimal_content = "Just a name\nNo other information"
        pdf_path = self.create_test_pdf(tmp_path, minimal_content)
        
        result = await self.manager.process_candidates(pdf_path.parent)
        assert len(result) == 1
        assert result[0]["skills"] == []  # Should handle missing fields gracefully
    
    async def test_candidate_batch_processing(self, tmp_path: Path) -> None:
        """Test processing multiple candidate PDFs."""
        # Create multiple test PDFs
        pdfs = []
        for i in range(3):
            content = f"""
            Candidate {i}
            Location: City {i}
            Skills: Python, Skill {i}
            Education: Degree {i}
            """
            pdfs.append(self.create_test_pdf(tmp_path, content))
        
        result = await self.manager.process_candidates(tmp_path)
        
        assert len(result) == 3
        assert len({c["id"] for c in result}) == 3  # Unique IDs
        assert all("Python" in c["skills"] for c in result)
    
    def teardown_method(self) -> None:
        """Clean up after test."""
        super().teardown_method() 