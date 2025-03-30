"""
validators.py

Shared codes for input validation in nncodec.
"""


import os
from typing import Any

def validate_type(variable: Any, name: str, expected_type: type) -> None:
    """Validate that variable is of the expected type."""
    if not isinstance(variable, expected_type):
        raise ValueError(f"{name} must be of type {expected_type.__name__}")


def validate_file_exists(file_path: str) -> None:
    """Validate that the given file path exists."""
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")