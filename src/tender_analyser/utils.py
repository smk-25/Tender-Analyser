"""
Utility functions for the Tender Analyser.
"""

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_env_config() -> dict:
    """Load configuration from environment variables."""
    from dotenv import load_dotenv
    
    env_path = get_project_root() / '.env'
    load_dotenv(env_path)
    
    return {
        'data_dir': os.getenv('DATA_DIR', 'data'),
        'log_dir': os.getenv('LOG_DIR', 'logs'),
        'debug': os.getenv('DEBUG', 'False').lower() == 'true'
    }


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
