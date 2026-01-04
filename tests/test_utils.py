"""
Tests for utility functions.
"""

from pathlib import Path
from tender_analyser.utils import get_project_root, ensure_directory
import tempfile
import shutil


def test_get_project_root():
    """Test getting project root."""
    root = get_project_root()
    assert root.exists()
    assert root.is_dir()


def test_ensure_directory():
    """Test ensuring directory exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test" / "nested" / "dir"
        ensure_directory(test_path)
        assert test_path.exists()
        assert test_path.is_dir()
