# Tender Analyser Documentation

## Overview

Tender Analyser is a Python-based tool designed to analyze tender documents and extract key information automatically.

## Architecture

The project follows a modular architecture:

1. **Analyser Module** (`analyser.py`): Core analysis logic
2. **Extractor Module** (`extractor.py`): Document data extraction
3. **CLI Module** (`cli.py`): Command-line interface
4. **Utils Module** (`utils.py`): Utility functions

## API Reference

### TenderAnalyser

Main class for analyzing tender documents.

#### Methods

- `__init__(config: Optional[Dict[str, Any]] = None)`: Initialize the analyser
- `analyze(file_path: Path) -> Dict[str, Any]`: Analyze a single document
- `batch_analyze(file_paths: List[Path]) -> List[Dict[str, Any]]`: Analyze multiple documents

### DataExtractor

Class for extracting data from various document formats.

#### Methods

- `extract_from_pdf(file_path: Path) -> Dict[str, Any]`: Extract from PDF
- `extract_from_excel(file_path: Path) -> Dict[str, Any]`: Extract from Excel
- `extract(file_path: Path) -> Dict[str, Any]`: Auto-detect and extract

## Usage Examples

### Basic Analysis

```python
from pathlib import Path
from tender_analyser import TenderAnalyser

analyser = TenderAnalyser()
result = analyser.analyze(Path("tender.pdf"))
```

### Batch Processing

```python
from pathlib import Path
from tender_analyser import TenderAnalyser

analyser = TenderAnalyser()
files = [Path("tender1.pdf"), Path("tender2.pdf")]
results = analyser.batch_analyze(files)
```

### Data Extraction

```python
from pathlib import Path
from tender_analyser import DataExtractor

extractor = DataExtractor()
data = extractor.extract(Path("tender.pdf"))
```

## Development Guide

### Adding New Document Formats

To add support for a new document format:

1. Add extraction method in `extractor.py`
2. Update the `extract()` method to handle the new format
3. Add tests for the new format

### Extending Analysis Capabilities

To add new analysis features:

1. Add methods to `TenderAnalyser` class
2. Update the `analyze()` method to include new features
3. Add corresponding tests

## Testing

The project uses pytest for testing. Tests are located in the `tests/` directory.

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_analyser.py

# With coverage
pytest --cov=src/tender_analyser
```

## Deployment

### Docker Deployment

```bash
docker-compose up -d
```

### Manual Deployment

```bash
pip install -e .
tender-analyser --help
```
