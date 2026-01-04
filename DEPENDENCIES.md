# Dependencies Documentation

This document provides information about the Python dependencies required for the Tender-Analyser project.

## Overview

The Tender-Analyser project has two main components with different dependency requirements:

1. **CLI Tool** (`src/tender_analyser/`) - A command-line tool for analyzing tender documents
2. **Streamlit Application** (`Summarizationcode.py`) - A web-based interactive UI for tender analysis

## Installation

### Quick Start

To install all required dependencies, run:

```bash
# Install production dependencies
pip install -r requirements.txt

# For development (includes testing and code quality tools)
pip install -r requirements-dev.txt

# Or install the package with all dependencies
pip install -e .
```

## Core Dependencies

### Document Processing
- **PyMuPDF** (==1.24.14) - PDF rendering and text extraction (imported as `fitz`)
- **pdfplumber** (>=0.10.0) - Advanced PDF table extraction
- **PyPDF2** (>=3.0.0) - Alternative PDF processing library
- **python-docx** (>=1.0.0) - Microsoft Word document generation and parsing
- **openpyxl** (>=3.1.0) - Excel file reading and writing

### Data Analysis
- **pandas** (>=2.0.0) - Data manipulation and analysis
- **numpy** (>=1.24.0) - Numerical computing support
- **nltk** (>=3.8.0) - Natural Language Toolkit for text processing

### AI/ML Integration
- **google-genai** (>=0.2.0) - Google Generative AI (Gemini) integration
- **streamlit** (>=1.28.0) - Web application framework for ML applications

### Utilities
- **click** (>=8.1.0) - Command-line interface creation
- **python-dotenv** (>=1.0.0) - Environment variable management
- **requests** (>=2.31.0) - HTTP library for API calls
- **beautifulsoup4** (>=4.12.0) - HTML/XML parsing

## Component-Specific Dependencies

### CLI Tool (`src/tender_analyser/`)
The CLI tool requires:
- click
- python-dotenv
- Standard library modules

### Streamlit Application (`Summarizationcode.py`)
The Streamlit application requires all dependencies listed in `requirements.txt`:
- streamlit
- pymupdf (fitz)
- pandas
- nltk
- google-genai
- pdfplumber
- openpyxl
- python-docx

## Dependency Files

The project maintains dependencies in multiple files:

1. **requirements.txt** - Production dependencies for the Streamlit application
2. **requirements-dev.txt** - Development dependencies (testing, linting, etc.)
3. **setup.py** - Package installation configuration with all dependencies
4. **pyproject.toml** - Modern Python packaging configuration

All files are synchronized to ensure consistency.

## Version Constraints

All dependencies include minimum version constraints to ensure compatibility:
- Versions are specified using `>=` to allow bug fixes and minor updates
- Major version constraints are included where breaking changes are known

## Security Considerations

All dependencies are:
- Actively maintained
- From trusted sources (PyPI)
- Regularly updated to address security vulnerabilities

To check for security vulnerabilities in dependencies:

```bash
pip install safety
safety check
```

## Troubleshooting

### Common Issues

1. **PyMuPDF Installation Fails**
   - Ensure you have build tools installed
   - On Linux: `sudo apt-get install build-essential`
   - On macOS: Install Xcode Command Line Tools

2. **NLTK Resources Missing**
   - Run the following to download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

3. **Google GenAI Authentication**
   - Set the `GOOGLE_API_KEY` environment variable
   - Or provide it through the Streamlit UI

## Updating Dependencies

To update dependencies to the latest compatible versions:

```bash
pip install --upgrade -r requirements.txt
```

To check for outdated packages:

```bash
pip list --outdated
```

## Development Dependencies

The `requirements-dev.txt` file includes additional tools for development:

### Testing
- pytest (>=7.4.0)
- pytest-cov (>=4.1.0)
- pytest-mock (>=3.11.0)

### Code Quality
- black (>=23.7.0) - Code formatter
- flake8 (>=6.1.0) - Style guide enforcement
- pylint (>=2.17.0) - Code analysis
- mypy (>=1.4.0) - Static type checking

### Development Tools
- ipython (>=8.14.0) - Enhanced Python shell
- jupyter (>=1.0.0) - Jupyter notebooks

## Notes

- Python 3.8+ is required
- Some dependencies have system-level requirements (e.g., C compilers for PyMuPDF)
- Virtual environment usage is strongly recommended

## License Compatibility

All dependencies are compatible with the MIT License used by this project.
