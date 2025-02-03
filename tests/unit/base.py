"""Base class for unit tests."""

import pytest


class BaseUnitTest:
    """Base class for unit tests."""
    
    def setup_method(self) -> None:
        """Set up test environment."""
        pass
        
    def teardown_method(self) -> None:
        """Clean up after test."""
        pass 