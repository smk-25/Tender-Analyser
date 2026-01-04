"""
Tests for the DataExtractor class.
"""

import pytest
import tempfile
from pathlib import Path
from tender_analyser.extractor import DataExtractor


def test_data_extractor_init():
    """Test DataExtractor initialization."""
    extractor = DataExtractor()
    assert extractor is not None


def test_extract_from_pdf_nonexistent_file():
    """Test extracting from non-existent PDF raises error."""
    extractor = DataExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract_from_pdf(Path("/nonexistent/file.pdf"))


def test_extract_from_excel_nonexistent_file():
    """Test extracting from non-existent Excel raises error."""
    extractor = DataExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract_from_excel(Path("/nonexistent/file.xlsx"))


def test_extract_unsupported_format():
    """Test extracting from unsupported format raises error."""
    extractor = DataExtractor()
    with pytest.raises(ValueError):
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        try:
            extractor.extract(temp_path)
        finally:
            temp_path.unlink()
