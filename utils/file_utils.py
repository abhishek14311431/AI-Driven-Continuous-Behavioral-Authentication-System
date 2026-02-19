"""
CSV and JSON file utility functions.
"""

import csv
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional


def read_csv_safe(file_path: str, default: Any = None) -> Optional[pd.DataFrame]:
    """Safely read CSV file, return default if error."""
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return default
    except Exception as e:
        print(f"Error reading CSV {file_path}: {e}")
        return default


def write_csv_safe(file_path: str, data: pd.DataFrame, append: bool = False):
    """Safely write CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if append and os.path.exists(file_path):
            existing = pd.read_csv(file_path)
            combined = pd.concat([existing, data], ignore_index=True)
            combined.to_csv(file_path, index=False)
        else:
            data.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error writing CSV {file_path}: {e}")


def read_json_safe(file_path: str, default: Any = None) -> Optional[Dict]:
    """Safely read JSON file, return default if error."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        print(f"Error reading JSON {file_path}: {e}")
        return default


def write_json_safe(file_path: str, data: Dict, indent: int = 2):
    """Safely write JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        print(f"Error writing JSON {file_path}: {e}")


def ensure_directory(file_path: str):
    """Ensure directory exists for file path."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def file_exists(file_path: str) -> bool:
    """Check if file exists."""
    return os.path.exists(file_path)


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except:
        return 0
