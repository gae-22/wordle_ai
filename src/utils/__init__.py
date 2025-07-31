"""Utilities package - Helper functions and common utilities.

This package contains utility functions, logging setup, and other
common functionality used throughout the WORDLE AI solver.
"""

from .helpers import (
    calculate_letter_frequencies,
    ensure_directory_exists,
    get_config_dir,
    normalize_word,
    setup_logging,
    validate_word,
)

__all__ = [
    "calculate_letter_frequencies",
    "ensure_directory_exists",
    "get_config_dir",
    "normalize_word",
    "setup_logging",
    "validate_word",
]
