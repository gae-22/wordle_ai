"""ML prediction engine for WORDLE solving.

This module provides machine learning-powered predictions for optimal
guess selection based on game state and historical patterns.
"""

import logging
from typing import Any

import numpy as np

from .. import MLModelError
from .features import FeatureExtractor
from .models import WordleMLModel

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Machine learning prediction engine for WORDLE solving.

    Uses trained models to predict the best guesses based on current
    game state and historical patterns.
    """

    def __init__(self) -> None:
        """Initialize prediction engine."""
        logger.debug("Initializing PredictionEngine")

        self._feature_extractor = FeatureExtractor()
        self._model: WordleMLModel | None = None
        self._is_trained = False

        # Initialize with simple heuristic model until training is complete
        self._use_heuristic = True

        logger.debug("PredictionEngine initialized")

    def predict_best_guesses(
        self,
        possible_words: list[str],
        previous_guesses: list[Any]
    ) -> dict[str, float]:
        """Predict scores for possible guesses.

        Args:
            possible_words: List of remaining possible words
            previous_guesses: List of previous guess results

        Returns:
            Dictionary mapping words to prediction scores (0-1)
        """
        if not possible_words:
            return {}

        logger.debug(f"Predicting scores for {len(possible_words)} words")

        try:
            # Always use our trained model if available
            if self._model is not None and self._is_trained:
                # Extract features for each word
                features_list = []
                valid_words = []

                for word in possible_words:
                    try:
                        word_features = self._feature_extractor.extract_features(
                            word, possible_words, previous_guesses
                        )
                        features_list.append(word_features)
                        valid_words.append(word)
                    except Exception as e:
                        logger.debug(f"Skipping word {word} due to feature extraction error: {e}")
                        continue

                if not features_list:
                    logger.warning("No valid features extracted, falling back to heuristic")
                    return self._heuristic_predictions(possible_words, previous_guesses)

                # Get model predictions
                features_array = np.array(features_list)
                predictions = self._model.predict(features_array)

                # Convert to dictionary
                word_scores = {}
                for word, score in zip(valid_words, predictions, strict=False):
                    word_scores[word] = float(np.clip(score, 0.0, 1.0))

                # Add any remaining words with heuristic scores
                for word in possible_words:
                    if word not in word_scores:
                        heuristic_scores = self._heuristic_predictions([word], previous_guesses)
                        word_scores[word] = heuristic_scores.get(word, 0.1)

                logger.debug(f"ML predictions generated for {len(word_scores)} words")
                return word_scores
            else:
                # Use heuristic predictions
                return self._heuristic_predictions(possible_words, previous_guesses)

        except Exception as e:
            logger.error(f"Prediction failed, falling back to heuristic: {e}")
            return self._heuristic_predictions(possible_words, previous_guesses)

    def score_guess(
        self,
        guess: str,
        possible_words: list[str],
        previous_guesses: list[Any]
    ) -> float:
        """Score a single guess.

        Args:
            guess: Word to score
            possible_words: List of remaining possible words
            previous_guesses: List of previous guess results

        Returns:
            Prediction score (0-1)
        """
        predictions = self.predict_best_guesses([guess], previous_guesses)
        return predictions.get(guess, 0.0)

    def train_models(self, training_data: list[dict[str, Any]]) -> None:
        """Train ML models on historical data.

        Args:
            training_data: List of training samples

        Raises:
            MLModelError: If training fails
        """
        logger.info(f"Starting ML model training with {len(training_data)} samples")

        if not training_data:
            logger.warning("No training data provided, using heuristic model")
            self._is_trained = True
            self._use_heuristic = True
            return

        try:
            # Import required models
            from .models import HeuristicModel, SimpleLinearModel

            # Extract features and targets from training data
            features_list = []
            targets_list = []

            for sample in training_data:
                try:
                    # Extract features for the word
                    word = sample.get('word', '')
                    if not word:
                        continue

                    # Create mock possible words and previous guesses for feature extraction
                    remaining_words = max(1, sample.get('remaining_words', 100))
                    mock_possible_words = [word] * min(remaining_words, 50)  # Simplified
                    mock_previous_guesses = []

                    # Extract features
                    word_features = self._feature_extractor.extract_features(
                        word, mock_possible_words, mock_previous_guesses
                    )

                    # Use success score as target
                    target_score = sample.get('success_score', sample.get('entropy', 0.5))

                    features_list.append(word_features)
                    targets_list.append(target_score)

                except Exception as e:
                    logger.debug(f"Skipping training sample: {e}")
                    continue

            if len(features_list) < 10:
                logger.warning(f"Insufficient training samples ({len(features_list)}), using heuristic model")
                self._model = HeuristicModel(adaptive_weights=True)
                self._is_trained = True
                self._use_heuristic = True
                return

            # Convert to numpy arrays
            features = np.array(features_list)
            targets = np.array(targets_list)

            logger.info(f"Training with {len(features)} samples, {features.shape[1]} features")

            # Train enhanced heuristic model with adaptive weights
            self._model = HeuristicModel(adaptive_weights=True)
            self._model.train(features, targets)

            # Try to train a simple linear model as backup
            try:
                linear_model = SimpleLinearModel()
                linear_model.train(features, targets)

                # Use linear model if we have enough data
                if len(features) > 100:
                    self._model = linear_model
                    self._use_heuristic = False
                    logger.info("Trained SimpleLinearModel successfully")
                else:
                    self._use_heuristic = True
                    logger.info("Using enhanced HeuristicModel with adaptive weights")

            except Exception as e:
                logger.warning(f"Linear model training failed, using heuristic: {e}")
                self._use_heuristic = True

            self._is_trained = True
            logger.info("ML model training completed successfully")

        except Exception as e:
            logger.error(f"ML model training failed: {e}")

            # Fallback to heuristic model
            try:
                from .models import HeuristicModel
                self._model = HeuristicModel(adaptive_weights=True)
                self._is_trained = True
                self._use_heuristic = True
                logger.info("Fallback to enhanced heuristic model")
            except Exception as fallback_error:
                logger.error(f"Fallback to heuristic also failed: {fallback_error}")
                raise MLModelError(f"Training failed: {e}") from e

    def _heuristic_predictions(
        self,
        possible_words: list[str],
        previous_guesses: list[Any]
    ) -> dict[str, float]:
        """Generate heuristic-based predictions.

        Args:
            possible_words: List of remaining possible words
            previous_guesses: List of previous guess results

        Returns:
            Dictionary mapping words to heuristic scores
        """
        word_scores = {}

        # Simple heuristic: prefer words with common letters
        common_letters = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
        letter_scores = {letter: (26 - i) / 26 for i, letter in enumerate(common_letters)}

        for word in possible_words:
            score = 0.0
            seen_letters = set()

            # Score based on letter frequency and uniqueness
            for letter in word.upper():
                if letter not in seen_letters:
                    score += letter_scores.get(letter, 0.1)
                    seen_letters.add(letter)
                else:
                    # Penalty for repeated letters
                    score -= 0.1

            # Normalize by word length
            score /= 5.0

            # Bonus for words with vowels
            vowels = set('AEIOU')
            vowel_count = sum(1 for letter in word.upper() if letter in vowels)
            score += vowel_count * 0.1

            # Penalty if too many attempts already made
            attempt_penalty = len(previous_guesses) * 0.05
            score = max(0.0, score - attempt_penalty)

            word_scores[word] = min(1.0, score)

        logger.debug(f"Generated heuristic scores for {len(word_scores)} words")
        return word_scores

    def get_model_info(self) -> dict[str, Any]:
        """Get information about current model state.

        Returns:
            Dictionary with model information
        """
        return {
            "is_trained": self._is_trained,
            "using_heuristic": self._use_heuristic,
            "model_type": "heuristic" if self._use_heuristic else "ml",
            "feature_count": self._feature_extractor.get_feature_count()
        }

    def reset_models(self) -> None:
        """Reset models to untrained state."""
        self._model = None
        self._is_trained = False
        self._use_heuristic = True
        logger.info("Models reset to untrained state")
