"""
Tests for the TenderAnalyser class.
"""

import pytest
from pathlib import Path
from tender_analyser.analyser import TenderAnalyser


def test_tender_analyser_init():
    """Test TenderAnalyser initialization."""
    analyser = TenderAnalyser()
    assert analyser is not None
    assert analyser.config == {}


def test_tender_analyser_init_with_config():
    """Test TenderAnalyser initialization with config."""
    config = {"key": "value"}
    analyser = TenderAnalyser(config=config)
    assert analyser.config == config


def test_analyze_nonexistent_file():
    """Test analyzing a non-existent file raises error."""
    analyser = TenderAnalyser()
    with pytest.raises(FileNotFoundError):
        analyser.analyze(Path("/nonexistent/file.pdf"))


def test_batch_analyze_empty_list():
    """Test batch analyzing with empty list."""
    analyser = TenderAnalyser()
    results = analyser.batch_analyze([])
    assert results == []
