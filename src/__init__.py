"""WORDLE AI Solver - A sophisticated TUI-based puzzle solver.

This package provides a terminal-based WORDLE solver that combines information
theory, entropy calculations, and machine learning to provide optimal guess
recommendations with adaptive learning capabilities.

Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "gae"
__email__ = "gaedev22@gmail.com"

from typing import Dict, List, Optional


# Core exceptions
class WordleAIException(Exception):
    """Base exception for all WORDLE AI errors."""

class WordListError(WordleAIException):
    """Errors related to word list management."""

class PatternError(WordleAIException):
    """Errors in pattern matching and analysis."""

class MLModelError(WordleAIException):
    """Machine learning model related errors."""

class EntropyCalculationError(WordleAIException):
    """Errors in entropy calculations."""

# Export main exceptions
__all__ = [
    "EntropyCalculationError",
    "MLModelError",
    "PatternError",
    "WordListError",
    "WordleAIException",
    "__author__",
    "__email__",
    "__version__",
]
