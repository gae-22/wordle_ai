"""Entropy calculations for WORDLE solving.

This module implements information theory-based calculations to determine
the optimal next guess by maximizing information gain (minimizing entropy).
"""

import logging
import math
from collections import defaultdict

from .. import EntropyCalculationError
from ..data.patterns import PatternMatcher

logger = logging.getLogger(__name__)


class EntropyCalculator:
    """Calculator for information entropy in WORDLE solving.

    Uses Shannon entropy to measure the information content of guesses
    and determine which guess provides the maximum information gain.
    """

    def __init__(self) -> None:
        """Initialize entropy calculator."""
        self._pattern_matcher = PatternMatcher()
        self._cache: dict[str, float] = {}
        logger.debug("EntropyCalculator initialized")

    def calculate_guess_entropy(self, guess: str, possible_words: list[str]) -> float:
        """Calculate entropy for a guess against possible words.

        Args:
            guess: The word being guessed
            possible_words: List of remaining possible words

        Returns:
            Shannon entropy value (bits of information)

        Raises:
            EntropyCalculationError: If calculation fails
        """
        if not possible_words:
            return 0.0

        if len(possible_words) == 1:
            return 0.0

        # Validate guess
        if not guess or len(guess) != 5 or not guess.isalpha():
            raise EntropyCalculationError(f"Invalid guess word: {guess!r}")

        cache_key = f"{guess}:{hash(tuple(sorted(possible_words)))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Count patterns that would result from this guess
            pattern_counts = self._count_patterns(guess, possible_words)

            if not pattern_counts:
                logger.warning(f"No valid patterns generated for guess '{guess}'")
                return 0.0

            # Calculate Shannon entropy
            total_words = sum(pattern_counts.values())
            if total_words == 0:
                return 0.0

            entropy = 0.0

            for count in pattern_counts.values():
                if count > 0:
                    probability = count / total_words
                    entropy -= probability * math.log2(probability)

            self._cache[cache_key] = entropy
            logger.debug(f"Entropy for '{guess}': {entropy:.3f} bits")

            return entropy

        except Exception as e:
            logger.error(f"Failed to calculate entropy for '{guess}': {e}")
            raise EntropyCalculationError(f"Entropy calculation failed: {e}") from e

    def find_best_guesses(
        self,
        possible_words: list[str],
        candidate_guesses: list[str],
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Find the best guesses by entropy.

        Args:
            possible_words: List of remaining possible words
            candidate_guesses: List of potential guesses to evaluate
            top_k: Number of top guesses to return

        Returns:
            List of (guess, entropy) tuples sorted by entropy (descending)

        Raises:
            EntropyCalculationError: If calculation fails
        """
        if not possible_words or not candidate_guesses:
            return []

        logger.debug(f"Evaluating {len(candidate_guesses)} candidate guesses")

        try:
            guess_entropies: list[tuple[str, float]] = []

            for guess in candidate_guesses:
                entropy = self.calculate_guess_entropy(guess, possible_words)
                guess_entropies.append((guess, entropy))

            # Sort by entropy (descending) and return top k
            guess_entropies.sort(key=lambda x: x[1], reverse=True)
            result = guess_entropies[:top_k]

            logger.debug(f"Top {len(result)} guesses by entropy: {result[:3]}")
            return result

        except Exception as e:
            logger.error(f"Failed to find best guesses: {e}")
            raise EntropyCalculationError(f"Best guess calculation failed: {e}") from e

    def calculate_expected_remaining_words(
        self,
        guess: str,
        possible_words: list[str]
    ) -> float:
        """Calculate expected number of remaining words after a guess.

        Args:
            guess: The word being guessed
            possible_words: List of remaining possible words

        Returns:
            Expected number of remaining words

        Raises:
            EntropyCalculationError: If calculation fails
        """
        if not possible_words:
            return 0.0

        try:
            pattern_counts = self._count_patterns(guess, possible_words)
            total_words = len(possible_words)

            expected_remaining = 0.0
            for count in pattern_counts.values():
                if count > 0:
                    probability = count / total_words
                    expected_remaining += probability * count

            return expected_remaining

        except Exception as e:
            logger.error(f"Failed to calculate expected remaining words: {e}")
            raise EntropyCalculationError(f"Expected remaining calculation failed: {e}") from e

    def _count_patterns(self, guess: str, possible_words: list[str]) -> dict[str, int]:
        """Count how many words would produce each pattern.

        Args:
            guess: The word being guessed
            possible_words: List of possible target words

        Returns:
            Dictionary mapping pattern strings to counts
        """
        pattern_counts: dict[str, int] = defaultdict(int)

        for word in possible_words:
            # Skip invalid words
            if not word or len(word) != 5 or not word.isalpha():
                logger.warning(f"Skipping invalid word: {word!r}")
                continue

            try:
                pattern = self._pattern_matcher.generate_pattern(guess, word)
                pattern_counts[pattern] += 1
            except Exception as e:
                logger.warning(f"Failed to generate pattern for guess '{guess}' vs word '{word}': {e}")
                continue

        return dict(pattern_counts)

    def calculate_positional_entropy(self, possible_words: list[str]) -> list[float]:
        """Calculate entropy for each letter position.

        Args:
            possible_words: List of remaining possible words

        Returns:
            List of entropy values for each position (0-4)

        Raises:
            EntropyCalculationError: If calculation fails
        """
        if not possible_words:
            return [0.0] * 5

        try:
            position_entropies = []

            for pos in range(5):
                # Count letter frequencies at this position
                letter_counts: dict[str, int] = defaultdict(int)

                for word in possible_words:
                    if pos < len(word):
                        letter_counts[word[pos]] += 1

                # Calculate entropy for this position
                total_words = len(possible_words)
                entropy = 0.0

                for count in letter_counts.values():
                    if count > 0:
                        probability = count / total_words
                        entropy -= probability * math.log2(probability)

                position_entropies.append(entropy)

            logger.debug(f"Positional entropies: {position_entropies}")
            return position_entropies

        except Exception as e:
            logger.error(f"Failed to calculate positional entropy: {e}")
            raise EntropyCalculationError(f"Positional entropy calculation failed: {e}") from e

    def get_letter_frequency_score(self, word: str, possible_words: list[str]) -> float:
        """Calculate letter frequency score for a word.

        Args:
            word: Word to score
            possible_words: List of possible words for frequency calculation

        Returns:
            Frequency score (higher is better)
        """
        if not possible_words:
            return 0.0

        try:
            # Count all letters in possible words
            letter_counts: dict[str, int] = defaultdict(int)
            total_letters = 0

            for possible_word in possible_words:
                for letter in possible_word:
                    letter_counts[letter] += 1
                    total_letters += 1

            # Calculate score for the word
            score = 0.0
            seen_letters: set[str] = set()

            for letter in word:
                if letter not in seen_letters:  # Only count each letter once
                    frequency = letter_counts[letter] / total_letters if total_letters > 0 else 0
                    score += frequency
                    seen_letters.add(letter)

            return score

        except Exception as e:
            logger.error(f"Failed to calculate frequency score for '{word}': {e}")
            return 0.0

    def clear_cache(self) -> None:
        """Clear the entropy calculation cache."""
        self._cache.clear()
        logger.debug("Entropy cache cleared")

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._cache),
            "cache_hits": getattr(self, "_cache_hits", 0),
            "cache_misses": getattr(self, "_cache_misses", 0)
        }
