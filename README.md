# Tender-Analyser

A tool for analyzing tender documents and extracting key information.

## Features

- Extract data from PDF and Excel tender documents
- Analyze tender content and structure
- Command-line interface for easy usage
- Batch processing capabilities
- Extensible architecture for custom analysis

## Installation

### Using pip (recommended)

```bash
# Clone the repository
git clone https://github.com/smk-25/Tender-Analyser.git
cd Tender-Analyser

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Using Docker

```bash
# Build the Docker image
docker-compose build

# Run the container
docker-compose run tender-analyser tender-analyser --help
```

## Quick Start

### Analyze a single tender document

```bash
tender-analyser analyze path/to/tender.pdf
```

### Analyze all documents in a directory

```bash
tender-analyser batch path/to/tender/directory -o results/
```

### Using as a Python library

```python
from pathlib import Path
from tender_analyser import TenderAnalyser

# Initialize the analyser
analyser = TenderAnalyser()

# Analyze a document
result = analyser.analyze(Path("tender.pdf"))
print(result)
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/tender_analyser --cov-report=html

# Run specific test file
pytest tests/test_analyser.py -v
```

### Code Quality

```bash
# Format code with black
black src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

## Project Structure

```
Tender-Analyser/
├── src/
│   └── tender_analyser/
│       ├── __init__.py
│       ├── analyser.py       # Main analysis logic
│       ├── extractor.py      # Data extraction from documents
│       ├── cli.py            # Command-line interface
│       └── utils.py          # Utility functions
├── tests/
│   ├── test_analyser.py
│   ├── test_extractor.py
│   └── test_utils.py
├── data/
│   ├── raw/                  # Raw tender documents
│   └── processed/            # Processed results
├── docs/                     # Documentation
├── logs/                     # Application logs
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD configuration
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── setup.py                  # Package setup
├── pyproject.toml           # Build configuration
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
└── README.md                # This file
```

## Configuration

Copy `.env.example` to `.env` and update the values:

```bash
cp .env.example .env
```

Configuration options:
- `DATA_DIR`: Directory for data files (default: `data`)
- `LOG_DIR`: Directory for log files (default: `logs`)
- `DEBUG`: Enable debug mode (default: `False`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on GitHub.