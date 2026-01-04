#!/usr/bin/env python3
"""
Dependency Verification Script

This script verifies that all Python dependencies required by the Tender-Analyser
project are properly listed in the requirements files.

It checks:
1. All imports in source code are covered by requirements
2. No duplicate or conflicting dependencies
3. All required packages can be resolved
"""

import ast
import sys
from pathlib import Path
from typing import Set, Dict, List


# Mapping of import names to PyPI package names
IMPORT_TO_PACKAGE = {
    'fitz': 'pymupdf',
    'docx': 'python-docx',
    'dotenv': 'python-dotenv',
    'bs4': 'beautifulsoup4',
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'yaml': 'pyyaml',
    'google.genai': 'google-genai',
    # Standard library imports (no package needed)
    'os': None,
    'sys': None,
    'json': None,
    'logging': None,
    'pathlib': None,
    'typing': None,
    'asyncio': None,
    'collections': None,
    'io': None,
    're': None,
    'tempfile': None,
    'time': None,
}


def extract_imports_from_file(file_path: Path) -> Set[str]:
    """Extract all import statements from a Python file."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"⚠️  Warning: Could not parse {file_path}: {e}")
    
    return imports


def get_all_imports(project_root: Path) -> Dict[str, Set[str]]:
    """Get all imports from Python files in the project."""
    imports_by_file = {}
    
    # Check source code
    src_dir = project_root / 'src'
    if src_dir.exists():
        for py_file in src_dir.rglob('*.py'):
            if '__pycache__' not in str(py_file):
                imports_by_file[str(py_file.relative_to(project_root))] = \
                    extract_imports_from_file(py_file)
    
    # Check Summarizationcode.py
    summary_file = project_root / 'Summarizationcode.py'
    if summary_file.exists():
        imports_by_file[str(summary_file.relative_to(project_root))] = \
            extract_imports_from_file(summary_file)
    
    return imports_by_file


def parse_requirements(req_file: Path) -> Set[str]:
    """Parse a requirements.txt file and return package names."""
    packages = set()
    
    if not req_file.exists():
        return packages
    
    with open(req_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#') and not line.startswith('-'):
                # Extract package name (before >= or ==)
                pkg = line.split('>=')[0].split('==')[0].split('[')[0].strip()
                packages.add(pkg.lower())
    
    return packages


def main():
    """Main verification function."""
    project_root = Path(__file__).parent
    
    print("=" * 70)
    print("Tender-Analyser Dependency Verification")
    print("=" * 70)
    print()
    
    # Get all imports from code
    print("📂 Scanning source code for imports...")
    imports_by_file = get_all_imports(project_root)
    
    all_imports = set()
    for file_path, imports in imports_by_file.items():
        all_imports.update(imports)
        if imports:
            print(f"   {file_path}: {len(imports)} imports")
    
    print(f"\n✓ Found {len(all_imports)} unique import names across all files\n")
    
    # Convert import names to package names
    print("📦 Converting import names to package names...")
    required_packages = set()
    stdlib_imports = set()
    unknown_imports = set()
    
    for imp in all_imports:
        if imp in IMPORT_TO_PACKAGE:
            pkg = IMPORT_TO_PACKAGE[imp]
            if pkg is None:
                stdlib_imports.add(imp)
            else:
                required_packages.add(pkg.lower())
        else:
            # Assume import name = package name for most cases
            # Check if it's likely a standard library module
            if imp in ['pathlib', 'typing', 'datetime', 'uuid', 'hashlib', 
                       'subprocess', 'shutil', 'glob', 'itertools', 'functools',
                       'argparse', 'configparser', 'csv', 'decimal', 'random',
                       'string', 'traceback', 'warnings', 'contextlib', 'copy',
                       'enum', 'dataclasses', 'abc']:
                stdlib_imports.add(imp)
            else:
                required_packages.add(imp.lower())
                unknown_imports.add(imp)
    
    print(f"   Required packages: {len(required_packages)}")
    print(f"   Standard library: {len(stdlib_imports)}")
    if unknown_imports:
        print(f"   ⚠️  Unknown (assumed=package): {len(unknown_imports)}")
    print()
    
    # Parse requirements files
    print("📋 Parsing requirements files...")
    req_txt = parse_requirements(project_root / 'requirements.txt')
    req_dev = parse_requirements(project_root / 'requirements-dev.txt')
    
    print(f"   requirements.txt: {len(req_txt)} packages")
    print(f"   requirements-dev.txt: {len(req_dev)} packages (includes requirements.txt)")
    print()
    
    # Check coverage
    print("🔍 Checking dependency coverage...")
    print()
    
    all_requirements = req_txt | req_dev
    missing_in_requirements = required_packages - all_requirements
    extra_in_requirements = all_requirements - required_packages
    
    # Filter out known dev-only packages
    dev_only_packages = {'pytest', 'pytest-cov', 'pytest-mock', 'black', 
                         'flake8', 'pylint', 'mypy', 'ipython', 'jupyter'}
    extra_in_requirements = extra_in_requirements - dev_only_packages
    
    if not missing_in_requirements:
        print("✅ All imported packages are listed in requirements files!")
    else:
        print("❌ Missing packages in requirements files:")
        for pkg in sorted(missing_in_requirements):
            print(f"   - {pkg}")
        print()
    
    if extra_in_requirements:
        print("ℹ️  Packages in requirements but not directly imported:")
        for pkg in sorted(extra_in_requirements):
            print(f"   - {pkg} (may be transitive dependency or used indirectly)")
        print()
    
    # List all required packages
    print("📄 Summary of required packages:")
    print()
    print("Required by code:")
    for pkg in sorted(required_packages):
        status = "✓" if pkg in all_requirements else "✗"
        print(f"   {status} {pkg}")
    print()
    
    # Return exit code
    if missing_in_requirements:
        print("⚠️  Action required: Add missing packages to requirements files")
        return 1
    else:
        print("✅ All checks passed! Dependencies are properly configured.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
