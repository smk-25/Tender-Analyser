from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tender-analyser",
    version="0.1.0",
    author="SMK-25",
    description="A tool for analyzing tender documents and extracting key information",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smk-25/Tender-Analyser",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "openpyxl>=3.1.0",
        "PyPDF2>=3.0.0",
        "pymupdf>=1.24.0",
        "pdfplumber>=0.10.0",
        "nltk>=3.9.0",
        "click>=8.1.0",
        "streamlit>=1.28.0",
        "google-genai>=0.2.0",
        "python-docx>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "tender-analyser=tender_analyser.cli:main",
        ],
    },
)
