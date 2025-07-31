"""Machine learning models for WORDLE solving.

This module defines ML model architectures and training procedures
for predicting optimal WORDLE guesses.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .. import MLModelError

logger = logging.getLogger(__name__)


class WordleMLModel(ABC):
    """Abstract base class for WORDLE ML models.

    Defines the interface that all WORDLE ML models must implement.
    """

    @abstractmethod
    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train the model on provided data.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            targets: Target values of shape (n_samples,)

        Raises:
            MLModelError: If training fails
        """

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on provided features.

        Args:
            features: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,)

        Raises:
            MLModelError: If prediction fails
        """

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save model to file.

        Args:
            filepath: Path to save model

        Raises:
            MLModelError: If saving fails
        """

    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load model from file.

        Args:
            filepath: Path to load model from

        Raises:
            MLModelError: If loading fails
        """

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary with model information
        """


class SimpleLinearModel(WordleMLModel):
    """Simple linear regression model for WORDLE guess scoring.

    A basic implementation for Phase 1, to be replaced with more
    sophisticated models in Phase 3.
    """

    def __init__(self) -> None:
        """Initialize simple linear model."""
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._is_trained: bool = False

        logger.debug("SimpleLinearModel initialized")

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train linear model using least squares.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            targets: Target values of shape (n_samples,)

        Raises:
            MLModelError: If training fails
        """
        try:
            if features.shape[0] != targets.shape[0]:
                raise MLModelError("Feature and target sample counts don't match")

            if features.shape[0] < features.shape[1]:
                raise MLModelError("Not enough samples for training")

            # Add bias column
            x_with_bias = np.column_stack([np.ones(features.shape[0]), features])

            # Solve normal equations: (X^T X)^-1 X^T y
            xtx = x_with_bias.T @ x_with_bias
            xty = x_with_bias.T @ targets

            # Add small regularization to prevent singular matrix
            reg_lambda = 1e-6
            xtx += reg_lambda * np.eye(xtx.shape[0])

            weights_with_bias = np.linalg.solve(xtx, xty)

            self._bias = weights_with_bias[0]
            self._weights = weights_with_bias[1:]
            self._is_trained = True

            logger.info(f"Model trained with {features.shape[0]} samples, {features.shape[1]} features")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise MLModelError(f"Training failed: {e}") from e

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using linear model.

        Args:
            features: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,)

        Raises:
            MLModelError: If prediction fails or model not trained
        """
        if not self._is_trained or self._weights is None:
            raise MLModelError("Model must be trained before making predictions")

        try:
            if features.shape[1] != self._weights.shape[0]:
                raise MLModelError(f"Feature dimension mismatch: expected {self._weights.shape[0]}, got {features.shape[1]}")

            predictions = features @ self._weights + self._bias

            # Clip predictions to [0, 1] range
            predictions = np.clip(predictions, 0.0, 1.0)

            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise MLModelError(f"Prediction failed: {e}") from e

    def save_model(self, filepath: str) -> None:
        """Save model parameters to file.

        Args:
            filepath: Path to save model

        Raises:
            MLModelError: If saving fails
        """
        if not self._is_trained:
            raise MLModelError("Cannot save untrained model")

        try:
            model_data = {
                'weights': self._weights,
                'bias': self._bias,
                'is_trained': self._is_trained,
                'model_type': 'SimpleLinearModel'
            }

            np.savez(filepath, **model_data)
            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise MLModelError(f"Failed to save model: {e}") from e

    def load_model(self, filepath: str) -> None:
        """Load model parameters from file.

        Args:
            filepath: Path to load model from

        Raises:
            MLModelError: If loading fails
        """
        try:
            model_data = np.load(filepath)

            if model_data['model_type'] != 'SimpleLinearModel':
                raise MLModelError(f"Model type mismatch: expected SimpleLinearModel, got {model_data['model_type']}")

            self._weights = model_data['weights']
            self._bias = float(model_data['bias'])
            self._is_trained = bool(model_data['is_trained'])

            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise MLModelError(f"Failed to load model: {e}") from e

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'SimpleLinearModel',
            'is_trained': self._is_trained,
            'num_features': len(self._weights) if self._weights is not None else 0,
            'bias': self._bias,
            'weights_norm': np.linalg.norm(self._weights) if self._weights is not None else 0.0
        }


class HeuristicModel(WordleMLModel):
    """Heuristic-based model for WORDLE guess scoring.

    Enhanced heuristic model with Phase 3 improvements including
    adaptive weights and performance optimization.
    """

    def __init__(self, adaptive_weights: bool = True) -> None:
        """Initialize heuristic model.

        Args:
            adaptive_weights: Whether to use adaptive letter weights
        """
        self._is_trained = True  # Always "trained" since it uses heuristics
        self.adaptive_weights = adaptive_weights

        # Base letter frequency weights (based on English letter frequencies)
        self._base_letter_weights = {
            'E': 0.127, 'T': 0.091, 'A': 0.082, 'O': 0.075, 'I': 0.070,
            'N': 0.067, 'S': 0.063, 'H': 0.061, 'R': 0.060, 'D': 0.043,
            'L': 0.040, 'C': 0.028, 'U': 0.028, 'M': 0.024, 'W': 0.024,
            'F': 0.022, 'G': 0.020, 'Y': 0.020, 'P': 0.019, 'B': 0.013,
            'V': 0.010, 'K': 0.008, 'J': 0.002, 'X': 0.002, 'Q': 0.001, 'Z': 0.001
        }

        # Initialize adaptive weights as copy of base weights
        self._letter_weights = self._base_letter_weights.copy()

        # Performance tracking for adaptive learning
        self._performance_history = []
        self._adaptation_count = 0

        logger.debug(f"HeuristicModel initialized with adaptive_weights={adaptive_weights}")

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Adapt heuristic weights based on training data.

        Args:
            features: Feature matrix (used for adaptive learning)
            targets: Target values (used for adaptive learning)
        """
        if not self.adaptive_weights:
            logger.debug("Heuristic model training (no-op - adaptive weights disabled)")
            return

        logger.debug("Adapting heuristic model weights based on training data")

        try:
            # Simple adaptive mechanism: adjust weights based on performance
            if len(features) > 0 and len(targets) > 0:
                # Calculate performance correlation with features
                for i, target in enumerate(targets):
                    # Update weights based on target performance
                    self._update_weights_from_sample(features[i] if len(features) > i else None, target)

                self._adaptation_count += 1
                logger.debug(f"Completed adaptation #{self._adaptation_count}")

        except Exception as e:
            logger.warning(f"Heuristic weight adaptation failed: {e}")

    def _update_weights_from_sample(self, features: np.ndarray | None, target: float) -> None:
        """Update letter weights based on a single sample.

        Args:
            features: Feature vector for the sample
            target: Target score for the sample
        """
        if features is None or len(features) < 5:
            return

        # Simple adaptation: boost/reduce weights based on performance
        learning_rate = 0.01

        # Assume first 26 features are letter frequencies
        letter_freq_features = features[:26] if len(features) >= 26 else features

        for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            if i < len(letter_freq_features) and letter in self._letter_weights:
                # Adjust weight based on target score and feature value
                adjustment = learning_rate * (target - 0.5) * letter_freq_features[i]
                self._letter_weights[letter] = max(0.001,
                    self._letter_weights[letter] + adjustment
                )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using enhanced heuristic rules.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
                     Expected features: [vowel_count, consonant_count, unique_letters,
                                        repeated_letters, common_letters_score, ...]

        Returns:
            Predictions of shape (n_samples,)
        """
        try:
            n_samples = features.shape[0]
            predictions = np.zeros(n_samples)

            for i in range(n_samples):
                sample_features = features[i]

                # Enhanced scoring with adaptive weights
                score = 0.0

                # Vowel/consonant balance (prefer 2-3 vowels)
                vowel_ratio = sample_features[0] if len(sample_features) > 0 else 0.4
                if 0.3 <= vowel_ratio <= 0.6:
                    score += 0.25
                elif 0.2 <= vowel_ratio <= 0.7:
                    score += 0.15
                else:
                    score += 0.05

                # Unique letters (prefer more unique letters)
                unique_ratio = sample_features[2] if len(sample_features) > 2 else 0.8
                score += unique_ratio * 0.35

                # Common letters score (using adaptive weights)
                if len(sample_features) > 4:
                    common_score = sample_features[4]
                    # Apply adaptive weight multiplier
                    weight_multiplier = sum(self._letter_weights.values()) / sum(self._base_letter_weights.values())
                    score += common_score * 0.25 * weight_multiplier

                # Position diversity bonus
                if len(sample_features) > 5:
                    position_diversity = sample_features[5]
                    score += position_diversity * 0.15

                # Repeated letters penalty (adaptive)
                if len(sample_features) > 3:
                    repeated_ratio = sample_features[3]
                    penalty_strength = 0.1 + (self._adaptation_count * 0.01)  # Increase with experience
                    score -= repeated_ratio * penalty_strength

                # Normalize to [0, 1]
                predictions[i] = min(1.0, max(0.0, score))

            return predictions

        except Exception as e:
            logger.error(f"Enhanced heuristic prediction failed: {e}")
            raise MLModelError(f"Enhanced heuristic prediction failed: {e}") from e

    def save_model(self, filepath: str) -> None:
        """Save heuristic model (just saves metadata).

        Args:
            filepath: Path to save model
        """
        try:
            model_data = {
                'model_type': 'HeuristicModel',
                'is_trained': True,
                'letter_weights': self._letter_weights
            }

            np.savez(filepath, **model_data)
            logger.info(f"Heuristic model saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save heuristic model: {e}")
            raise MLModelError(f"Failed to save heuristic model: {e}") from e

    def load_model(self, filepath: str) -> None:
        """Load heuristic model (just loads metadata).

        Args:
            filepath: Path to load model from
        """
        try:
            model_data = np.load(filepath, allow_pickle=True)

            if str(model_data['model_type']) != 'HeuristicModel':
                raise MLModelError("Model type mismatch: expected HeuristicModel")

            # Load letter weights if available
            if 'letter_weights' in model_data:
                self._letter_weights = model_data['letter_weights'].item()

            logger.info(f"Heuristic model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load heuristic model: {e}")
            raise MLModelError(f"Failed to load heuristic model: {e}") from e

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the enhanced heuristic model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'HeuristicModel',
            'is_trained': True,
            'adaptive_weights': self.adaptive_weights,
            'adaptation_count': self._adaptation_count,
            'uses_heuristics': True,
            'num_letter_weights': len(self._letter_weights),
            'weight_deviation': self._calculate_weight_deviation(),
            'description': 'Enhanced rule-based heuristic model with adaptive learning (Phase 3)'
        }

    def _calculate_weight_deviation(self) -> float:
        """Calculate how much weights have deviated from base weights.

        Returns:
            Average deviation from base weights
        """
        if not self.adaptive_weights:
            return 0.0

        deviations = []
        for letter in self._base_letter_weights:
            if letter in self._letter_weights:
                base_weight = self._base_letter_weights[letter]
                current_weight = self._letter_weights[letter]
                deviation = abs(current_weight - base_weight) / base_weight
                deviations.append(deviation)

        return np.mean(deviations) if deviations else 0.0
