"""UI package - Terminal user interface components.

This package provides Rich-based TUI components for interactive
WORDLE solving with beautiful terminal displays.
"""

from .display import WordleDisplay
from .input import InputHandler

__all__ = [
    "InputHandler",
    "WordleDisplay",
]
