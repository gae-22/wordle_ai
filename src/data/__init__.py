"""Data package - Word lists and pattern matching.

This package handles WORDLE word lists, pattern generation, and word filtering
based on game feedback.
"""

from .patterns import PatternMatcher
from .words import WordListManager

__all__ = [
    "PatternMatcher",
    "WordListManager",
]
