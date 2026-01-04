"""
Tender Analyser - A tool for analyzing tender documents and extracting key information.
"""

__version__ = "0.1.0"

from .analyser import TenderAnalyser
from .extractor import DataExtractor

__all__ = ["TenderAnalyser", "DataExtractor", "__version__"]
