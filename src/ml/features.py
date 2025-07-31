"""Feature extraction for WORDLE ML models.

This module extracts relevant features from game state, word characteristics,
and historical patterns for use in machine learning models.
"""

import logging
from collections import Counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts features for ML models from WORDLE game state.

    Converts game state and word characteristics into numerical features
    that can be used by machine learning models.
    """

    def __init__(self) -> None:
        """Initialize feature extractor."""
        self._feature_names = [
            # Word characteristics
            "vowel_count", "consonant_count", "unique_letters", "repeated_letters",
            "common_letters_score", "letter_position_score",

            # Game state features
            "attempt_number", "remaining_words_log", "entropy_estimate",

            # Historical features
            "letter_frequency_score", "position_frequency_score",

            # Pattern features
            "green_positions", "yellow_positions", "gray_letters_count"
        ]

        # Common letter frequencies (approximate English frequencies)
        self._letter_frequencies = {
            'E': 0.127, 'T': 0.091, 'A': 0.082, 'O': 0.075, 'I': 0.070,
            'N': 0.067, 'S': 0.063, 'H': 0.061, 'R': 0.060, 'D': 0.043,
            'L': 0.040, 'C': 0.028, 'U': 0.028, 'M': 0.024, 'W': 0.024,
            'F': 0.022, 'G': 0.020, 'Y': 0.020, 'P': 0.019, 'B': 0.013,
            'V': 0.010, 'K': 0.008, 'J': 0.002, 'X': 0.002, 'Q': 0.001, 'Z': 0.001
        }

        logger.debug("FeatureExtractor initialized")

    def extract_features(
        self,
        word: str,
        possible_words: list[str],
        previous_guesses: list[Any]
    ) -> np.ndarray:
        """Extract features for a word given current game state.

        Args:
            word: Word to extract features for
            possible_words: List of remaining possible words
            previous_guesses: List of previous guess results

        Returns:
            Feature vector as numpy array
        """
        word = word.upper()
        features = []

        # Word characteristics
        features.extend(self._extract_word_features(word))

        # Game state features
        features.extend(self._extract_game_state_features(word, possible_words, previous_guesses))

        # Historical features
        features.extend(self._extract_historical_features(word, possible_words))

        # Pattern features
        features.extend(self._extract_pattern_features(word, previous_guesses))

        return np.array(features, dtype=np.float32)

    def _extract_word_features(self, word: str) -> list[float]:
        """Extract features related to word characteristics.

        Args:
            word: Word to analyze

        Returns:
            List of word-related features
        """
        vowels = set('AEIOU')
        consonants = set('BCDFGHJKLMNPQRSTVWXYZ')

        # Basic counts
        vowel_count = sum(1 for letter in word if letter in vowels)
        consonant_count = sum(1 for letter in word if letter in consonants)
        unique_letters = len(set(word))
        repeated_letters = 5 - unique_letters

        # Common letters score
        common_letters_score = sum(
            self._letter_frequencies.get(letter, 0.0) for letter in set(word)
        ) / len(set(word))

        # Letter position score (some letters are more common in certain positions)
        position_scores = {
            0: {'S': 0.3, 'C': 0.2, 'B': 0.2, 'T': 0.2, 'P': 0.2, 'A': 0.15, 'F': 0.15},
            1: {'A': 0.3, 'O': 0.25, 'R': 0.2, 'E': 0.2, 'I': 0.15, 'L': 0.15},
            2: {'A': 0.2, 'I': 0.2, 'O': 0.15, 'U': 0.15, 'R': 0.15, 'N': 0.1},
            3: {'E': 0.25, 'S': 0.2, 'A': 0.15, 'R': 0.15, 'N': 0.15, 'L': 0.1},
            4: {'E': 0.3, 'Y': 0.25, 'R': 0.2, 'S': 0.15, 'T': 0.1, 'A': 0.1}
        }

        letter_position_score = 0.0
        for i, letter in enumerate(word):
            letter_position_score += position_scores.get(i, {}).get(letter, 0.05)
        letter_position_score /= 5.0

        return [
            vowel_count / 5.0,
            consonant_count / 5.0,
            unique_letters / 5.0,
            repeated_letters / 5.0,
            common_letters_score,
            letter_position_score
        ]

    def _extract_game_state_features(
        self,
        word: str,
        possible_words: list[str],
        previous_guesses: list[Any]
    ) -> list[float]:
        """Extract features related to current game state.

        Args:
            word: Word to analyze
            possible_words: List of remaining possible words
            previous_guesses: List of previous guess results

        Returns:
            List of game state features
        """
        attempt_number = len(previous_guesses) + 1
        remaining_words = len(possible_words)

        # Log scale for remaining words
        remaining_words_log = np.log(max(1, remaining_words)) / np.log(1000)

        # Rough entropy estimate (will be replaced with actual entropy calculation)
        entropy_estimate = 0.5 if remaining_words > 10 else 0.2

        return [
            attempt_number / 6.0,  # Normalize by max attempts
            remaining_words_log,
            entropy_estimate
        ]

    def _extract_historical_features(
        self,
        word: str,
        possible_words: list[str]
    ) -> list[float]:
        """Extract features based on historical patterns.

        Args:
            word: Word to analyze
            possible_words: List of remaining possible words

        Returns:
            List of historical features
        """
        if not possible_words:
            return [0.0, 0.0]

        # Letter frequency in remaining words
        letter_counts = Counter()
        position_counts = [Counter() for _ in range(5)]

        for possible_word in possible_words:
            for letter in possible_word:
                letter_counts[letter] += 1
            for i, letter in enumerate(possible_word):
                position_counts[i][letter] += 1

        total_letters = sum(letter_counts.values())

        # Score word based on letter frequencies
        letter_frequency_score = 0.0
        for letter in set(word):
            freq = letter_counts.get(letter, 0) / max(1, total_letters)
            letter_frequency_score += freq
        letter_frequency_score /= len(set(word))

        # Score word based on position frequencies
        position_frequency_score = 0.0
        for i, letter in enumerate(word):
            pos_total = sum(position_counts[i].values())
            freq = position_counts[i].get(letter, 0) / max(1, pos_total)
            position_frequency_score += freq
        position_frequency_score /= 5.0

        return [letter_frequency_score, position_frequency_score]

    def _extract_pattern_features(self, word: str, previous_guesses: list[Any]) -> list[float]:
        """Extract features based on previous guess patterns.

        Args:
            word: Word to analyze
            previous_guesses: List of previous guess results

        Returns:
            List of pattern-based features
        """
        if not previous_guesses:
            return [0.0, 0.0, 0.0]

        green_positions = 0
        yellow_positions = 0
        gray_letters = set()

        # Analyze previous guesses (simplified - assumes guess results have pattern info)
        for guess_result in previous_guesses:
            # This is a placeholder - actual implementation would depend on
            # the structure of guess results
            if hasattr(guess_result, 'pattern'):
                pattern = guess_result.pattern
                green_positions += pattern.count('G')
                yellow_positions += pattern.count('Y')

                if hasattr(guess_result, 'guess'):
                    guess_word = guess_result.guess
                    for _i, (letter, color) in enumerate(zip(guess_word, pattern, strict=False)):
                        if color == 'X':
                            gray_letters.add(letter)

        # Check if current word conflicts with known constraints
        gray_conflicts = sum(1 for letter in word if letter in gray_letters)

        return [
            green_positions / max(1, len(previous_guesses) * 5),
            yellow_positions / max(1, len(previous_guesses) * 5),
            gray_conflicts / 5.0
        ]

    def get_feature_names(self) -> list[str]:
        """Get list of feature names.

        Returns:
            List of feature names in order
        """
        return self._feature_names.copy()

    def get_feature_count(self) -> int:
        """Get number of features extracted.

        Returns:
            Number of features
        """
        return len(self._feature_names)

    def batch_extract_features(
        self,
        words: list[str],
        possible_words: list[str],
        previous_guesses: list[Any]
    ) -> np.ndarray:
        """Extract features for multiple words at once.

        Args:
            words: List of words to extract features for
            possible_words: List of remaining possible words
            previous_guesses: List of previous guess results

        Returns:
            Feature matrix with shape (len(words), num_features)
        """
        features_list = []

        for word in words:
            features = self.extract_features(word, possible_words, previous_guesses)
            features_list.append(features)

        return np.vstack(features_list) if features_list else np.array([])

    def validate_features(self, features: np.ndarray) -> bool:
        """Validate extracted features.

        Args:
            features: Feature vector to validate

        Returns:
            True if features are valid
        """
        if features.shape[0] != len(self._feature_names):
            logger.warning(f"Feature count mismatch: expected {len(self._feature_names)}, got {features.shape[0]}")
            return False

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning("Features contain NaN or infinite values")
            return False

        return True
