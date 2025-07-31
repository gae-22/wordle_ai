"""Helper functions and utilities for WORDLE AI solver.

This module provides common utility functions, logging setup, and
other helper functionality used throughout the application.
"""

import json
import logging
import logging.handlers
import sys
from collections import Counter
from collections import Counter as CounterType
from pathlib import Path

from .. import WordleAIException


def validate_word(word: str) -> None:
    """Validate that a word is suitable for WORDLE.

    Args:
        word: Word to validate

    Raises:
        WordleAIException: If word is invalid
    """
    if not word:
        raise WordleAIException("Word cannot be empty")

    if len(word) != 5:
        raise WordleAIException(f"Word must be exactly 5 letters: '{word}' has {len(word)}")

    if not word.isalpha():
        raise WordleAIException(f"Word must contain only letters: '{word}'")


def normalize_word(word: str) -> str:
    """Normalize a word to standard format.

    Args:
        word: Word to normalize

    Returns:
        Normalized word (uppercase, stripped)

    Raises:
        WordleAIException: If word is invalid after normalization
    """
    normalized = word.strip().upper()
    validate_word(normalized)
    return normalized


def setup_logging(level: str = "INFO", log_file: str | None = None, quiet: bool = False) -> None:
    """Set up structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        quiet: Suppress console output if True
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler (unless quiet mode)
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            if not quiet:
                root_logger.warning(f"Failed to setup file logging: {e}")

    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def calculate_letter_frequencies(words: list[str]) -> dict[str, float]:
    """Calculate letter frequencies from a list of words.

    Args:
        words: List of words to analyze

    Returns:
        Dictionary mapping letters to their frequencies
    """
    if not words:
        return {}

    letter_counts: CounterType[str] = Counter()
    total_letters = 0

    for word in words:
        normalized_word = normalize_word(word)
        for letter in normalized_word:
            letter_counts[letter] += 1
            total_letters += 1

    # Convert counts to frequencies
    frequencies = {}
    for letter, count in letter_counts.items():
        frequencies[letter] = count / total_letters if total_letters > 0 else 0.0

    return frequencies


def calculate_positional_frequencies(words: list[str]) -> list[dict[str, float]]:
    """Calculate letter frequencies for each position.

    Args:
        words: List of words to analyze

    Returns:
        List of dictionaries, one for each position (0-4)
    """
    if not words:
        return [{} for _ in range(5)]

    position_counts = [Counter() for _ in range(5)]

    for word in words:
        normalized_word = normalize_word(word)
        for i, letter in enumerate(normalized_word):
            if i < 5:  # Safety check
                position_counts[i][letter] += 1

    # Convert to frequencies
    position_frequencies = []
    for i in range(5):
        total = sum(position_counts[i].values())
        frequencies = {}
        for letter, count in position_counts[i].items():
            frequencies[letter] = count / total if total > 0 else 0.0
        position_frequencies.append(frequencies)

    return position_frequencies


def get_config_dir() -> Path:
    """Get application configuration directory.

    Returns:
        Path to configuration directory
    """
    # Use XDG base directory specification on Unix systems
    if sys.platform.startswith('win'):
        config_dir = Path.home() / 'AppData' / 'Local' / 'WordleAI'
    else:
        xdg_config_home = Path.home() / '.config'
        config_dir = xdg_config_home / 'wordle-ai'

    return config_dir


def ensure_directory_exists(directory: Path) -> None:
    """Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path to create

    Raises:
        WordleAIException: If directory cannot be created
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise WordleAIException(f"Failed to create directory {directory}: {e}") from e


def load_json_file(filepath: Path) -> dict:
    """Load data from a JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded data dictionary

    Raises:
        WordleAIException: If file cannot be loaded
    """
    try:
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise WordleAIException(f"File not found: {filepath}") from None
    except json.JSONDecodeError as e:
        raise WordleAIException(f"Invalid JSON in {filepath}: {e}") from e
    except Exception as e:
        raise WordleAIException(f"Failed to load {filepath}: {e}") from e


def save_json_file(data: dict, filepath: Path) -> None:
    """Save data to a JSON file.

    Args:
        data: Data to save
        filepath: Path to save file

    Raises:
        WordleAIException: If file cannot be saved
    """
    try:
        ensure_directory_exists(filepath.parent)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise WordleAIException(f"Failed to save {filepath}: {e}") from e


def find_common_letters(words: list[str], top_k: int = 10) -> list[str]:
    """Find the most common letters across all words.

    Args:
        words: List of words to analyze
        top_k: Number of top letters to return

    Returns:
        List of most common letters in descending order
    """
    frequencies = calculate_letter_frequencies(words)

    # Sort by frequency (descending)
    sorted_letters = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

    return [letter for letter, _ in sorted_letters[:top_k]]


def find_unique_letter_words(words: list[str]) -> list[str]:
    """Find words with all unique letters.

    Args:
        words: List of words to filter

    Returns:
        List of words with no repeated letters
    """
    unique_words = []

    for word in words:
        try:
            normalized = normalize_word(word)
            if len(set(normalized)) == 5:  # All letters unique
                unique_words.append(normalized)
        except WordleAIException:
            continue  # Skip invalid words

    return unique_words


def calculate_word_diversity(words: list[str]) -> float:
    """Calculate diversity score for a set of words.

    Args:
        words: List of words to analyze

    Returns:
        Diversity score (0-1, higher is more diverse)
    """
    if not words:
        return 0.0

    # Count unique letters
    all_letters: set[str] = set()
    for word in words:
        try:
            normalized = normalize_word(word)
            all_letters.update(normalized)
        except WordleAIException:
            continue

    # Diversity is ratio of unique letters to maximum possible (26)
    diversity = len(all_letters) / 26.0
    return min(1.0, diversity)


def benchmark_function(func, *args, **kwargs):
    """Benchmark a function's execution time.

    Args:
        func: Function to benchmark
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Tuple of (result, execution_time_seconds)
    """
    import time

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    execution_time = end_time - start_time
    return result, execution_time


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to specified length.

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix
