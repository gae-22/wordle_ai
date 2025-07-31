# WORDLE AI Solver - Copilot Development Instructions

## Project Mission & Context

**PRIMARY OBJECTIVE**: Create a sophisticated TUI (Terminal User Interface) based WORDLE solver that combines information theory, entropy calculations, and machine learning to provide optimal guess recommendations with adaptive learning capabilities.

**CORE PHILOSOPHY**: This is NOT a web application or GUI - it's a rich, interactive terminal application that provides an immersive command-line experience for solving WORDLE puzzles with mathematical precision and intelligent pattern recognition through ML models.

## Technical Stack

-   **Language**: Python 3.10+
-   **Package Manager**: uv
-   **TUI Framework**: Rich
-   **ML Libraries**: scikit-learn, numpy, pandas, torch
-   **Analytics**: matplotlib, seaborn, scipy
-   **Development Tools**: Ruff, MyPy
-   **Performance**: psutil, memory optimization tools

## Development Environment Commands

-   **Script Execution**: `uv run python {filename}`
-   **Tool Execution**: `uv tool run {toolname}`
-   **Package Installation**: `uv add {package}`
-   **Development Dependencies**: `uv add --dev {package}`

## Code Style & Quality Guidelines

### Python Code Standards

#### 1. Type Annotations

**Do:**

-   Use type hints for all function parameters and return values
-   Use `typing` module for complex types (Union, Optional, List, Dict)
-   Use generic types where appropriate (T, K, V)
-   Document type aliases for complex types

**Do not:**

-   Skip type annotations for public APIs
-   Use `Any` type unless absolutely necessary
-   Mix typed and untyped code in the same module

```python
# ✅ Do
from typing import List, Optional, Dict, Union
def calculate_entropy(words: List[str], patterns: Dict[str, int]) -> float:
    """Calculate entropy for word list."""
    pass

# ❌ Do not
def calculate_entropy(words, patterns):
    pass
```

#### 2. Docstrings

**Do:**

-   Follow Google-style docstrings consistently
-   Document all public functions, classes, and modules
-   Include Args, Returns, and Raises sections
-   Provide usage examples for complex functions

**Do not:**

-   Skip docstrings for public APIs
-   Use inconsistent docstring formats
-   Write vague or incomplete descriptions

```python
# ✅ Do
def filter_words_by_pattern(words: List[str], pattern: str) -> List[str]:
    """Filter words based on WORDLE pattern feedback.

    Args:
        words: List of candidate words to filter
        pattern: Pattern string using G(green), Y(yellow), X(gray)

    Returns:
        Filtered list of words matching the pattern

    Raises:
        PatternError: If pattern format is invalid

    Example:
        >>> filter_words_by_pattern(['AROSE', 'TIGER'], 'GYXXX')
        ['TIGER']
    """
    pass

# ❌ Do not
def filter_words_by_pattern(words: List[str], pattern: str) -> List[str]:
    """Filter words."""
    pass
```

#### 3. Error Handling

**Do:**

-   Create specific exception classes for different error types
-   Use try-except blocks for expected failure scenarios
-   Log errors with context information
-   Fail fast with meaningful error messages

**Do not:**

-   Use bare `except:` clauses
-   Ignore exceptions silently
-   Raise generic Exception for specific errors

```python
# ✅ Do
class WordListError(WordleAIException):
    """Raised when word list operations fail."""
    pass

def load_word_list(filepath: str) -> List[str]:
    try:
        with open(filepath, 'r') as f:
            words = [line.strip().upper() for line in f]
        if not words:
            raise WordListError(f"Empty word list in {filepath}")
        return words
    except FileNotFoundError:
        raise WordListError(f"Word list file not found: {filepath}")

# ❌ Do not
def load_word_list(filepath: str) -> List[str]:
    try:
        with open(filepath, 'r') as f:
            return [line.strip().upper() for line in f]
    except:
        return []
```

#### 4. Code Organization

**Do:**

-   Follow single responsibility principle
-   Use clear, descriptive names for functions and variables
-   Group related functionality into modules
-   Keep functions focused and small (< 50 lines)

**Do not:**

-   Create monolithic functions or classes
-   Use misleading or abbreviated names
-   Mix different levels of abstraction

#### 5. Logging

**Do:**

-   Use structured logging with consistent format
-   Include contextual information (game state, performance metrics)
-   Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
-   Log performance-critical operations

**Do not:**

-   Use print() statements for debugging
-   Log sensitive information
-   Create excessive log noise

```python
# ✅ Do
import logging
logger = logging.getLogger(__name__)

def calculate_best_guess(words: List[str], game_state: GameState) -> str:
    logger.info(f"Calculating best guess from {len(words)} candidates")
    start_time = time.time()

    result = entropy_calculator.find_optimal_guess(words, game_state)

    duration = time.time() - start_time
    logger.info(f"Best guess calculation completed in {duration:.3f}s: {result}")
    return result

# ❌ Do not
def calculate_best_guess(words: List[str], game_state: GameState) -> str:
    print(f"Calculating from {len(words)} words")
    result = entropy_calculator.find_optimal_guess(words, game_state)
    print(f"Result: {result}")
    return result
```

#### 6. Performance

**Do:**

-   Profile code before optimizing
-   Use appropriate data structures (sets for membership, deque for queues)
-   Cache expensive computations when possible
-   Monitor memory usage for large datasets

**Do not:**

-   Optimize prematurely without profiling
-   Use inefficient algorithms for large datasets
-   Ignore memory leaks or excessive memory usage

### Project Structure

```
src/
├── __init__.py          # Version and package info
├── main.py              # Entry point with CLI interface
├── solver/              # Core solver logic
│   ├── __init__.py
│   ├── engine.py        # Main solving algorithm
│   ├── entropy.py       # Information theory calculations
│   └── strategy.py      # Guessing strategies
├── data/                # Word lists and game data
│   ├── __init__.py
│   ├── words.py         # Word list management
│   └── patterns.py      # Pattern matching logic
├── ml/                  # Machine learning components
│   ├── __init__.py
│   ├── features.py      # Feature engineering
│   ├── models.py        # ML model definitions
│   ├── neural_models.py # Neural network implementations
│   ├── adaptive_learning.py # Adaptive learning algorithms
│   ├── performance_optimization.py # Performance optimization tools
│   ├── prediction.py    # Prediction engine
│   └── training.py      # Model training logic
├── analytics/           # Advanced analytics (Phase 4) ✅
│   ├── __init__.py
│   ├── statistics.py    # Statistical analysis tools
│   ├── strategy_comparison.py # Strategy comparison
│   ├── difficulty_prediction.py # Word difficulty prediction
│   └── game_theory.py   # Game theory optimization
├── ui/                  # Terminal user interface
│   ├── __init__.py
│   ├── display.py       # Rich-based UI components
│   └── input.py         # User input handling
└── utils/               # Utility functions
    ├── __init__.py
    └── helpers.py       # Common helper functions
```

## Core Algorithms & Implementation

### Information Theory Approach

-   Calculate entropy for each possible guess
-   Use pattern frequency analysis to determine optimal moves
-   Implement Bayesian inference for probability updates

### Machine Learning Integration

-   **Feature Engineering**: Extract features from word patterns, letter frequencies, and game states
-   **Predictive Models**: Train models to predict word difficulty and optimal strategies
-   **Adaptive Learning**: Learn from game outcomes to improve future performance
-   **Pattern Recognition**: Use ML to identify complex patterns in word relationships

### Key Components to Implement

1. **Word List Manager**: Handle valid words and answer sets ✅
2. **Pattern Analyzer**: Process WORDLE feedback (🟩🟨⬜) ✅
3. **Entropy Calculator**: Compute information gain for each guess ✅
4. **ML Feature Extractor**: Convert game state into ML features ✅
5. **Prediction Engine**: ML-powered guess scoring and ranking ✅
6. **Strategy Engine**: Combine entropy and ML predictions for optimal decisions ✅
7. **TUI Controller**: Rich-based interactive interface ✅
8. **Analytics Suite**: Statistical analysis and strategy comparison ✅
9. **Neural Networks**: Deep learning models for pattern recognition ✅
10. **Adaptive Learning**: Dynamic strategy optimization ✅

## Performance Requirements

-   Solve puzzles in under 6 guesses (average ≤ 4)
-   Response time < 1 second per guess recommendation
-   Memory usage < 100MB for word processing

## Implementation Priorities

1. **Phase 1**: Core solver engine with entropy calculations ✅
2. **Phase 2**: Pattern matching and word filtering ✅
3. **Phase 3**: Machine learning model development and training ✅
4. **Phase 4**: Advanced analytics and strategy optimization ✅
5. **Phase 5**: Performance optimization and advanced features 📋

## Code Quality Automation

-   Use Ruff for code formatting and linting
-   MyPy for static type checking
-   Comprehensive error handling with exception chaining
-   Performance profiling and optimization tools
-   Advanced analytics and benchmarking capabilities

## AI Assistant Guidelines

**Do:**

-   Start with comprehensive type definitions for all new modules
-   Write detailed docstrings before implementing complex algorithms
-   Consider edge cases like empty word lists, invalid patterns
-   Design for extensibility and maintainability from the start
-   Add structured logging with context for debugging
-   Profile and benchmark performance-critical code paths
-   Use appropriate data structures for the specific use case
-   Implement proper error handling with custom exceptions

**Do not:**

-   Skip type annotations or use `Any` without justification
-   Implement features without proper documentation
-   Ignore error handling or edge cases
-   Use print statements instead of proper logging
-   Optimize prematurely without profiling first
-   Mix business logic with UI code
-   Create functions that do too many things

## Domain-Specific Knowledge

-   WORDLE uses 5-letter words from a curated list
-   Feedback system: Green (correct position), Yellow (wrong position), Gray (not in word)
-   Common starting words: AROSE, ADIEU, AUDIO (high vowel content)
-   Advanced strategy: Consider letter frequency and position statistics

## Development Workflow

### Feature Development Process

**Do:**

1. **Analysis Phase**: Research existing solutions, gather clear requirements, understand problem constraints
2. **Implementation Phase**: Follow established patterns, write clean code, implement incrementally
3. **Documentation Phase**: Update README, add inline comments, create usage examples
4. **Review Phase**: Self-review code, run all checks, validate against requirements

**Do not:**

-   Skip the analysis phase and jump straight to coding
-   Design monolithic solutions without considering modularity
-   Implement large features in single commits
-   Forget to update documentation when changing APIs
-   Submit code without running linters and formatters

### Git Workflow

**Do:**

-   Use descriptive feature branch names (feat/entropy-optimization, fix/pattern-matching-bug)
-   Write atomic commits that represent single logical changes
-   Follow Conventional Commits specification for all commit messages
-   Run `uv run ruff check` and `uv run mypy` before pushing
-   Keep branches focused on single features or fixes

**Do not:**

-   Work directly on main/master branch
-   Create massive commits with multiple unrelated changes
-   Use vague commit messages like "fix stuff" or "update code"
-   Push code without running quality checks
-   Mix refactoring with feature development in same commit
-   Let feature branches become stale or too large

### Commit Message Convention

Follow the **Conventional Commits** specification for consistent, semantic commit messages:

#### Format Structure

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Types

-   **feat** ✨: A new feature for the user
-   **fix** 🐛: A bug fix
-   **docs** 📚: Documentation only changes
-   **style** 💎: Changes that do not affect the meaning of the code
-   **refactor** ♻️: Code change that neither fixes a bug nor adds a feature
-   **perf** ⚡: Performance improvements
-   **build** 📦: Changes that affect the build system or external dependencies
-   **ci** 🤖: Changes to CI configuration files and scripts
-   **chore** 🔧: Other changes that don't modify src files

#### Examples

```
feat ✨(entropy): add Shannon entropy calculation for guess optimization
fix 🐛(patterns): resolve incorrect pattern matching for yellow letters
docs 📚(readme): update installation instructions for uv package manager
refactor ♻️(solver): extract strategy logic into separate modules
perf ⚡(ml): optimize feature extraction for faster predictions
build 📦(deps): upgrade scikit-learn to version 1.3.0
ci 🤖(github): add automated testing workflow
chore(gitignore): add __pycache__ to ignore list
```

#### Breaking Changes

For breaking changes, add `!` after the type/scope:

```
feat(api)!: change solver interface to return structured results
```

#### Guidelines

-   Use imperative mood in the description ("add" not "added" or "adds")
-   Don't capitalize the first letter of description
-   No period at the end of description
-   Limit first line to 72 characters
-   Include issue number in footer if applicable: `Closes #123`

### Performance Optimization Guidelines

**Do:**

-   Profile before optimizing
-   Focus on algorithmic improvements first
-   Use appropriate data structures (sets for membership, deque for queues)
-   Cache expensive computations when possible
-   Monitor memory usage for large datasets

**Do not:**

-   Optimize prematurely without profiling
-   Use inefficient algorithms for large datasets
-   Ignore memory leaks or excessive memory usage

-   Profile before optimizing
-   Focus on algorithmic improvements first
-   Use appropriate data structures for the task
-   Cache expensive computations when possible
-   Monitor memory usage during development

## Error Handling Strategy

### Exception Hierarchy

```python
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
```

### Logging Configuration

-   Use structured logging with JSON format
-   Include context information (game state, performance metrics)
-   Different log levels for development vs production
-   Performance logging for optimization insights
