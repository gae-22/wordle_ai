"""Solver package - Core WORDLE solving algorithms.

This package contains the main solving engine that combines information theory,
entropy calculations, and machine learning for optimal guess recommendations.
"""

from .engine import WordleSolver
from .entropy import EntropyCalculator
from .strategy import SolvingStrategy

__all__ = [
    "EntropyCalculator",
    "SolvingStrategy",
    "WordleSolver",
]
