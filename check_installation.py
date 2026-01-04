#!/usr/bin/env python3
"""
Simple installation verification script for Tender-Analyser.

This script checks if all required packages are installed and importable.
"""

import sys
from importlib import import_module


def check_package(package_name, import_name=None):
    """Check if a package can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        import_module(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def main():
    print("=" * 70)
    print("Tender-Analyser Installation Verification")
    print("=" * 70)
    print()
    
    # Packages for Streamlit application
    streamlit_packages = [
        ('streamlit', 'streamlit'),
        ('PyMuPDF', 'fitz'),
        ('pandas', 'pandas'),
        ('nltk', 'nltk'),
        ('google-genai', 'google.genai'),
        ('pdfplumber', 'pdfplumber'),
        ('openpyxl', 'openpyxl'),
        ('python-docx', 'docx'),
    ]
    
    # Packages for CLI tool
    cli_packages = [
        ('click', 'click'),
        ('python-dotenv', 'dotenv'),
        ('requests', 'requests'),
        ('beautifulsoup4', 'bs4'),
        ('PyPDF2', 'PyPDF2'),
        ('numpy', 'numpy'),
    ]
    
    print("Checking Streamlit Application Dependencies:")
    print("-" * 70)
    streamlit_missing = []
    for pkg_name, import_name in streamlit_packages:
        success, error = check_package(pkg_name, import_name)
        status = "✓" if success else "✗"
        print(f"  {status} {pkg_name:20s} ", end="")
        if success:
            print("(installed)")
        else:
            print(f"(NOT FOUND)")
            streamlit_missing.append(pkg_name)
    
    print()
    print("Checking CLI Tool Dependencies:")
    print("-" * 70)
    cli_missing = []
    for pkg_name, import_name in cli_packages:
        success, error = check_package(pkg_name, import_name)
        status = "✓" if success else "✗"
        print(f"  {status} {pkg_name:20s} ", end="")
        if success:
            print("(installed)")
        else:
            print(f"(NOT FOUND)")
            cli_missing.append(pkg_name)
    
    print()
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    
    if not streamlit_missing and not cli_missing:
        print("✅ All packages are installed correctly!")
        print()
        print("You can now:")
        print("  • Run the CLI tool: tender-analyser --help")
        print("  • Run the Streamlit app: streamlit run Summarizationcode.py")
        return 0
    
    if streamlit_missing:
        print(f"❌ Missing {len(streamlit_missing)} Streamlit app package(s):")
        for pkg in streamlit_missing:
            print(f"   - {pkg}")
        print()
        print("To install Streamlit app dependencies:")
        print("  pip install -r requirements.txt")
        print()
    
    if cli_missing:
        print(f"⚠️  Missing {len(cli_missing)} CLI tool package(s):")
        for pkg in cli_missing:
            print(f"   - {pkg}")
        print()
        print("To install CLI tool dependencies:")
        print("  pip install -e .")
        print()
    
    return 1


if __name__ == '__main__':
    sys.exit(main())
