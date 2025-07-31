"""Solving strategies for WORDLE.

This module implements different solving strategies that can be used
individually or combined for optimal performance.
"""

import logging
import math
from enum import Enum
from typing import Any

from .. import WordleAIException
from ..ml.prediction import PredictionEngine
from .entropy import EntropyCalculator

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available solving strategies."""
    ENTROPY = "entropy"
    ML = "ml"
    HYBRID = "hybrid"


class SolvingStrategy:
    """Main strategy coordinator for WORDLE solving.

    Combines different approaches (entropy, ML, hybrid) to determine
    the best next guess based on the current game state.
    """

    def __init__(
        self,
        strategy_type: str,
        entropy_calculator: EntropyCalculator,
        prediction_engine: PredictionEngine
    ) -> None:
        """Initialize solving strategy.

        Args:
            strategy_type: Type of strategy to use
            entropy_calculator: Entropy calculation engine
            prediction_engine: ML prediction engine

        Raises:
            WordleAIException: If strategy type is invalid
        """
        try:
            self.strategy_type = StrategyType(strategy_type)
        except ValueError:
            raise WordleAIException(f"Invalid strategy type: {strategy_type}") from None

        self._entropy_calculator = entropy_calculator
        self._prediction_engine = prediction_engine

        # Strategy weights for hybrid approach
        self._entropy_weight = 0.6
        self._ml_weight = 0.4

        logger.info(f"Initialized strategy: {self.strategy_type.value}")

    def get_best_guess(
        self,
        possible_words: list[str],
        previous_guesses: list[Any]
    ) -> str:
        """Get the best next guess based on current strategy.

        Args:
            possible_words: List of remaining possible words
            previous_guesses: List of previous guess results

        Returns:
            Best guess word

        Raises:
            WordleAIException: If no valid guess can be determined
        """
        if not possible_words:
            raise WordleAIException("No possible words remaining")

        if len(possible_words) == 1:
            return possible_words[0]

        logger.debug(f"Finding best guess from {len(possible_words)} possibilities")

        try:
            if self.strategy_type == StrategyType.ENTROPY:
                return self._get_entropy_based_guess(possible_words, previous_guesses)

            elif self.strategy_type == StrategyType.ML:
                return self._get_ml_based_guess(possible_words, previous_guesses)

            elif self.strategy_type == StrategyType.HYBRID:
                return self._get_hybrid_guess(possible_words, previous_guesses)

            else:
                raise WordleAIException(f"Unhandled strategy type: {self.strategy_type}")

        except Exception as e:
            logger.error(f"Failed to get best guess: {e}")
            # Fallback to first available word
            return possible_words[0]

    def _get_entropy_based_guess(
        self,
        possible_words: list[str],
        previous_guesses: list[Any]
    ) -> str:
        """Get best guess using entropy maximization.

        Args:
            possible_words: List of remaining possible words
            previous_guesses: List of previous guess results

        Returns:
            Best guess by entropy
        """
        logger.debug("Using entropy-based strategy")

        # Use all valid words as candidates for first guess
        # Use remaining words for subsequent guesses to guarantee solvability
        candidate_guesses = self._get_initial_candidates() if not previous_guesses else possible_words

        # Find best guesses by entropy
        best_guesses = self._entropy_calculator.find_best_guesses(
            possible_words, candidate_guesses, top_k=1
        )

        if not best_guesses:
            logger.warning("No entropy-based guesses found, using first possible word")
            return possible_words[0]

        best_guess = best_guesses[0][0]
        entropy = best_guesses[0][1]

        logger.info(f"Entropy-based guess: {best_guess} (entropy: {entropy:.3f})")
        return best_guess

    def _get_ml_based_guess(
        self,
        possible_words: list[str],
        previous_guesses: list[Any]
    ) -> str:
        """Get best guess using machine learning predictions.

        Args:
            possible_words: List of remaining possible words
            previous_guesses: List of previous guess results

        Returns:
            Best guess by ML prediction
        """
        logger.debug("Using ML-based strategy")

        # Get ML predictions for possible words
        predictions = self._prediction_engine.predict_best_guesses(
            possible_words, previous_guesses
        )

        if not predictions:
            logger.warning("No ML predictions available, using first possible word")
            return possible_words[0]

        # Return highest scoring word
        best_guess = max(predictions, key=predictions.get)
        score = predictions[best_guess]

        logger.info(f"ML-based guess: {best_guess} (score: {score:.3f})")
        return best_guess

    def _get_hybrid_guess(
        self,
        possible_words: list[str],
        previous_guesses: list[Any]
    ) -> str:
        """Get best guess using hybrid entropy + ML approach.

        Args:
            possible_words: List of remaining possible words
            previous_guesses: List of previous guess results

        Returns:
            Best guess by combined score
        """
        logger.debug("Using hybrid strategy")

        # Get candidates for scoring
        candidate_guesses = self._get_initial_candidates() if not previous_guesses else possible_words

        # Limit candidates for performance
        if len(candidate_guesses) > 100:
            candidate_guesses = candidate_guesses[:100]

        best_guess = None
        best_score = -1.0

        # Score each candidate using both methods
        for candidate in candidate_guesses:
            # Calculate entropy score (normalized 0-1)
            entropy = self._entropy_calculator.calculate_guess_entropy(
                candidate, possible_words
            )
            max_entropy = math.log2(len(possible_words)) if len(possible_words) > 1 else 1.0
            entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0

            # Get ML score (already 0-1)
            ml_score = self._prediction_engine.score_guess(
                candidate, possible_words, previous_guesses
            )

            # Combined weighted score
            combined_score = (
                self._entropy_weight * entropy_score +
                self._ml_weight * ml_score
            )

            if combined_score > best_score:
                best_score = combined_score
                best_guess = candidate

        if not best_guess:
            logger.warning("No hybrid guess found, using first possible word")
            return possible_words[0]

        logger.info(f"Hybrid guess: {best_guess} (score: {best_score:.3f})")
        return best_guess

    def _get_initial_candidates(self) -> list[str]:
        """Get initial candidate words for first guess.

        Returns:
            List of good starting words
        """
        # Common good starting words with high vowel content
        return [
            "AROSE", "ADIEU", "AUDIO", "OUNCE", "MEDIA",
            "TRAIN", "SLATE", "CRANE", "IRATE", "TRACE",
            "CRATE", "STARE", "RAISE", "ARISE", "LEARN"
        ]

    def update_strategy_weights(self, entropy_weight: float, ml_weight: float) -> None:
        """Update strategy weights for hybrid approach.

        Args:
            entropy_weight: Weight for entropy component (0-1)
            ml_weight: Weight for ML component (0-1)

        Raises:
            WordleAIException: If weights are invalid
        """
        if not (0 <= entropy_weight <= 1 and 0 <= ml_weight <= 1):
            raise WordleAIException("Strategy weights must be between 0 and 1")

        if abs(entropy_weight + ml_weight - 1.0) > 0.01:
            raise WordleAIException("Strategy weights must sum to 1.0")

        self._entropy_weight = entropy_weight
        self._ml_weight = ml_weight

        logger.info(f"Updated strategy weights: entropy={entropy_weight:.2f}, ml={ml_weight:.2f}")

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about current strategy configuration.

        Returns:
            Dictionary with strategy information
        """
        return {
            "strategy_type": self.strategy_type.value,
            "entropy_weight": self._entropy_weight,
            "ml_weight": self._ml_weight,
            "description": self._get_strategy_description()
        }

    def _get_strategy_description(self) -> str:
        """Get description of current strategy.

        Returns:
            Human-readable strategy description
        """
        if self.strategy_type == StrategyType.ENTROPY:
            return "Pure information theory approach using Shannon entropy"
        elif self.strategy_type == StrategyType.ML:
            return "Machine learning predictions based on historical patterns"
        elif self.strategy_type == StrategyType.HYBRID:
            return f"Hybrid approach: {self._entropy_weight:.0%} entropy, {self._ml_weight:.0%} ML"
        else:
            return "Unknown strategy"
