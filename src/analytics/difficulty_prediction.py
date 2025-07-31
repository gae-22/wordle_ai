"""Word difficulty prediction for WORDLE AI.

This module provides machine learning-based prediction of word difficulty
and analysis of factors that make words harder or easier to solve.
"""

import logging
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from .. import MLModelError, WordleAIException
from ..ml.features import FeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class DifficultyFeatures:
    """Features that influence word difficulty."""
    letter_frequency_score: float
    vowel_count: int
    consonant_clusters: int
    double_letters: int
    uncommon_letters: int
    position_frequency_score: float
    word_length: int
    entropy_estimate: float


@dataclass
class DifficultyPrediction:
    """Word difficulty prediction result."""
    word: str
    predicted_attempts: float
    difficulty_category: str
    confidence: float
    contributing_factors: dict[str, float]


@dataclass
class DifficultyAnalysis:
    """Comprehensive difficulty analysis results."""
    model_performance: dict[str, float]
    feature_importance: dict[str, float]
    difficulty_distribution: dict[str, int]
    hardest_words: list[tuple[str, float]]
    easiest_words: list[tuple[str, float]]
    insights: list[str]


class DifficultyPredictor:
    """Machine learning-based word difficulty predictor.

    Analyzes and predicts the difficulty of WORDLE words based on
    various linguistic and statistical features.
    """

    def __init__(self) -> None:
        """Initialize difficulty predictor."""
        logger.debug("Initializing DifficultyPredictor")

        self._feature_extractor = FeatureExtractor()
        self._model: RandomForestRegressor | None = None
        self._scaler: StandardScaler | None = None
        self._is_trained = False

        # Feature weights for heuristic scoring
        self._feature_weights = {
            'letter_frequency': 0.25,
            'vowel_count': 0.15,
            'consonant_clusters': 0.20,
            'double_letters': 0.10,
            'uncommon_letters': 0.15,
            'position_frequency': 0.15
        }

        logger.debug("DifficultyPredictor initialized")

    def train_model(
        self,
        training_data: list[tuple[str, int]],
        validation_split: float = 0.2,
        model_type: str = "random_forest"
    ) -> dict[str, float]:
        """Train difficulty prediction model.

        Args:
            training_data: List of (word, attempts) tuples
            validation_split: Fraction of data to use for validation
            model_type: Type of model to train ("random_forest" or "gradient_boosting")

        Returns:
            Dictionary with training metrics

        Raises:
            MLModelError: If training fails
        """
        if len(training_data) < 10:
            raise MLModelError("Insufficient training data (need at least 10 samples)")

        logger.info(f"Training difficulty prediction model with {len(training_data)} samples")

        try:
            # Extract features and targets
            X, y = self._prepare_training_data(training_data)

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )

            # Scale features
            self._scaler = StandardScaler()
            X_train_scaled = self._scaler.fit_transform(X_train)
            X_val_scaled = self._scaler.transform(X_val)

            # Initialize model
            if model_type == "random_forest":
                self._model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "gradient_boosting":
                self._model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            else:
                raise MLModelError(f"Unknown model type: {model_type}")

            # Train model
            self._model.fit(X_train_scaled, y_train)

            # Validate model
            train_pred = self._model.predict(X_train_scaled)
            val_pred = self._model.predict(X_val_scaled)

            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)

            # Cross-validation
            cv_scores = cross_val_score(
                self._model, X_train_scaled, y_train, cv=5, scoring='r2'
            )

            self._is_trained = True

            metrics = {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'cv_mean_r2': cv_scores.mean(),
                'cv_std_r2': cv_scores.std(),
                'training_samples': len(training_data)
            }

            logger.info(f"Model training completed. Validation R²: {val_r2:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise MLModelError(f"Model training failed: {e}") from e

    def predict_difficulty(self, word: str) -> DifficultyPrediction:
        """Predict difficulty for a single word.

        Args:
            word: Word to predict difficulty for

        Returns:
            DifficultyPrediction with prediction results

        Raises:
            WordleAIException: If prediction fails
        """
        if len(word) != 5 or not word.isalpha():
            raise WordleAIException("Word must be 5 letters and alphabetic")

        word = word.upper()

        try:
            if self._is_trained and self._model is not None:
                # Use trained model
                features = self._extract_features(word)
                X = np.array([list(features.values())]).reshape(1, -1)
                X_scaled = self._scaler.transform(X)

                predicted_attempts = self._model.predict(X_scaled)[0]

                # Calculate confidence based on feature similarity to training data
                confidence = self._calculate_prediction_confidence(features)

                # Get feature importance for this prediction
                feature_names = list(features.keys())
                feature_importance = dict(zip(feature_names, self._model.feature_importances_, strict=False))

            else:
                # Use heuristic scoring
                predicted_attempts = self._heuristic_difficulty_score(word)
                confidence = 0.7  # Moderate confidence for heuristic
                feature_importance = self._feature_weights.copy()

            # Determine difficulty category
            difficulty_category = self._categorize_difficulty(predicted_attempts)

            return DifficultyPrediction(
                word=word,
                predicted_attempts=predicted_attempts,
                difficulty_category=difficulty_category,
                confidence=confidence,
                contributing_factors=feature_importance
            )

        except Exception as e:
            logger.error(f"Difficulty prediction failed for {word}: {e}")
            raise WordleAIException(f"Difficulty prediction failed: {e}") from e

    def predict_batch(self, words: list[str]) -> list[DifficultyPrediction]:
        """Predict difficulty for multiple words.

        Args:
            words: List of words to predict

        Returns:
            List of DifficultyPrediction results
        """
        predictions = []
        for word in words:
            try:
                prediction = self.predict_difficulty(word)
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Failed to predict difficulty for {word}: {e}")

        return predictions

    def analyze_difficulty_patterns(
        self,
        test_data: list[tuple[str, int]]
    ) -> DifficultyAnalysis:
        """Analyze difficulty patterns in the test data.

        Args:
            test_data: List of (word, actual_attempts) tuples

        Returns:
            DifficultyAnalysis with comprehensive analysis

        Raises:
            WordleAIException: If analysis fails
        """
        if not test_data:
            raise WordleAIException("No test data provided for analysis")

        logger.info(f"Analyzing difficulty patterns for {len(test_data)} words")

        try:
            # Generate predictions for all test words
            predictions = []
            actual_attempts = []

            for word, attempts in test_data:
                try:
                    pred = self.predict_difficulty(word)
                    predictions.append(pred.predicted_attempts)
                    actual_attempts.append(attempts)
                except Exception as e:
                    logger.warning(f"Failed to predict {word}: {e}")

            # Calculate model performance if we have predictions
            model_performance = {}
            if predictions and actual_attempts:
                mse = mean_squared_error(actual_attempts, predictions)
                r2 = r2_score(actual_attempts, predictions)
                mae = np.mean(np.abs(np.array(actual_attempts) - np.array(predictions)))

                model_performance = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'mae': mae,
                    'r2': r2
                }

            # Feature importance analysis
            feature_importance = {}
            if self._is_trained and self._model is not None:
                feature_names = [
                    'letter_frequency_score', 'vowel_count', 'consonant_clusters',
                    'double_letters', 'uncommon_letters', 'position_frequency_score',
                    'word_length', 'entropy_estimate'
                ]
                feature_importance = dict(zip(feature_names, self._model.feature_importances_, strict=False))

            # Difficulty distribution
            difficulty_distribution = self._analyze_difficulty_distribution(test_data)

            # Find hardest and easiest words
            word_difficulty_pairs = [(word, attempts) for word, attempts in test_data]
            hardest_words = sorted(word_difficulty_pairs, key=lambda x: x[1], reverse=True)[:10]
            easiest_words = sorted(word_difficulty_pairs, key=lambda x: x[1])[:10]

            # Generate insights
            insights = self._generate_difficulty_insights(
                test_data, feature_importance, difficulty_distribution
            )

            return DifficultyAnalysis(
                model_performance=model_performance,
                feature_importance=feature_importance,
                difficulty_distribution=difficulty_distribution,
                hardest_words=hardest_words,
                easiest_words=easiest_words,
                insights=insights
            )

        except Exception as e:
            logger.error(f"Difficulty analysis failed: {e}")
            raise WordleAIException(f"Difficulty analysis failed: {e}") from e

    def _prepare_training_data(
        self,
        training_data: list[tuple[str, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model training.

        Args:
            training_data: List of (word, attempts) tuples

        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        features_list = []
        targets = []

        for word, attempts in training_data:
            try:
                features = self._extract_features(word)
                features_list.append(list(features.values()))
                targets.append(attempts)
            except Exception as e:
                logger.warning(f"Failed to extract features for {word}: {e}")

        return np.array(features_list), np.array(targets)

    def _extract_features(self, word: str) -> dict[str, float]:
        """Extract difficulty-related features from a word.

        Args:
            word: Word to extract features from

        Returns:
            Dictionary of features
        """
        word = word.upper()

        # Basic features
        vowels = set('AEIOU')
        vowel_count = sum(1 for c in word if c in vowels)

        # Letter frequency (based on English letter frequency)
        english_freq = {
            'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75,
            'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78,
            'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97,
            'P': 1.93, 'B': 1.29, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15,
            'Q': 0.10, 'Z': 0.07
        }

        letter_freq_score = sum(english_freq.get(c, 0.01) for c in word) / len(word)

        # Count double letters
        double_letters = sum(1 for i in range(len(word) - 1) if word[i] == word[i + 1])

        # Count uncommon letters (frequency < 2%)
        uncommon_letters = sum(1 for c in word if english_freq.get(c, 0) < 2.0)

        # Count consonant clusters
        consonant_clusters = 0
        in_cluster = False
        for c in word:
            if c not in vowels:
                if in_cluster:
                    consonant_clusters += 1
                in_cluster = True
            else:
                in_cluster = False

        # Position-based frequency (simplified)
        position_freq_score = 0
        position_weights = [1.0, 0.8, 0.6, 0.8, 1.0]  # First and last positions more important
        for i, c in enumerate(word):
            position_freq_score += english_freq.get(c, 0.01) * position_weights[i]
        position_freq_score /= len(word)

        # Estimate entropy (simplified)
        unique_letters = len(set(word))
        entropy_estimate = unique_letters / len(word)

        return {
            'letter_frequency_score': letter_freq_score,
            'vowel_count': vowel_count,
            'consonant_clusters': consonant_clusters,
            'double_letters': double_letters,
            'uncommon_letters': uncommon_letters,
            'position_frequency_score': position_freq_score,
            'word_length': len(word),
            'entropy_estimate': entropy_estimate
        }

    def _heuristic_difficulty_score(self, word: str) -> float:
        """Calculate heuristic difficulty score for a word.

        Args:
            word: Word to score

        Returns:
            Predicted number of attempts (heuristic)
        """
        features = self._extract_features(word)

        # Base difficulty
        base_score = 3.5

        # Adjust based on features
        if features['vowel_count'] <= 1:
            base_score += 0.5
        elif features['vowel_count'] >= 4:
            base_score -= 0.3

        if features['uncommon_letters'] >= 2:
            base_score += 0.7

        if features['consonant_clusters'] >= 2:
            base_score += 0.4

        if features['double_letters'] >= 1:
            base_score += 0.3

        if features['letter_frequency_score'] < 3.0:
            base_score += 0.6

        return max(1.0, min(6.0, base_score))

    def _categorize_difficulty(self, predicted_attempts: float) -> str:
        """Categorize difficulty based on predicted attempts.

        Args:
            predicted_attempts: Predicted number of attempts

        Returns:
            Difficulty category string
        """
        if predicted_attempts <= 2.5:
            return "Easy"
        elif predicted_attempts <= 3.5:
            return "Medium"
        elif predicted_attempts <= 4.5:
            return "Hard"
        else:
            return "Very Hard"

    def _calculate_prediction_confidence(self, features: dict[str, float]) -> float:
        """Calculate confidence in prediction based on features.

        Args:
            features: Extracted features

        Returns:
            Confidence score (0-1)
        """
        # Simplified confidence calculation
        # In practice, this would use training data statistics
        base_confidence = 0.8

        # Lower confidence for unusual feature combinations
        if features['vowel_count'] == 0 or features['vowel_count'] == 5:
            base_confidence -= 0.2

        if features['uncommon_letters'] >= 3:
            base_confidence -= 0.1

        return max(0.1, min(1.0, base_confidence))

    def _analyze_difficulty_distribution(
        self,
        test_data: list[tuple[str, int]]
    ) -> dict[str, int]:
        """Analyze the distribution of word difficulties.

        Args:
            test_data: List of (word, attempts) tuples

        Returns:
            Dictionary mapping difficulty categories to counts
        """
        distribution = {"Easy": 0, "Medium": 0, "Hard": 0, "Very Hard": 0}

        for _, attempts in test_data:
            category = self._categorize_difficulty(attempts)
            distribution[category] += 1

        return distribution

    def _generate_difficulty_insights(
        self,
        test_data: list[tuple[str, int]],
        feature_importance: dict[str, float],
        difficulty_distribution: dict[str, int]
    ) -> list[str]:
        """Generate insights about word difficulty patterns.

        Args:
            test_data: Test data
            feature_importance: Feature importance scores
            difficulty_distribution: Difficulty distribution

        Returns:
            List of insight strings
        """
        insights = []

        total_words = len(test_data)
        avg_attempts = np.mean([attempts for _, attempts in test_data])

        insights.append(f"Average difficulty: {avg_attempts:.1f} attempts across {total_words} words")

        # Distribution insights
        hard_percentage = (difficulty_distribution["Hard"] + difficulty_distribution["Very Hard"]) / total_words * 100
        if hard_percentage > 30:
            insights.append(f"High difficulty dataset: {hard_percentage:.0f}% of words are hard or very hard")

        # Feature importance insights
        if feature_importance:
            most_important = max(feature_importance.keys(), key=lambda k: feature_importance[k])
            insights.append(f"Most important difficulty factor: {most_important.replace('_', ' ')}")

        # Specific pattern insights
        vowel_light_words = [word for word, _ in test_data if sum(1 for c in word if c in 'AEIOU') <= 1]
        if len(vowel_light_words) > total_words * 0.1:
            avg_vowel_light = np.mean([attempts for word, attempts in test_data if word in vowel_light_words])
            insights.append(f"Words with ≤1 vowel average {avg_vowel_light:.1f} attempts (typically harder)")

        return insights

    def save_model(self, filepath: str) -> None:
        """Save trained model to file.

        Args:
            filepath: Path to save model

        Raises:
            MLModelError: If save fails
        """
        if not self._is_trained or self._model is None:
            raise MLModelError("No trained model to save")

        try:
            model_data = {
                'model': self._model,
                'scaler': self._scaler,
                'feature_weights': self._feature_weights,
                'is_trained': self._is_trained
            }

            joblib.dump(model_data, filepath)
            logger.info(f"Difficulty prediction model saved to {filepath}")

        except Exception as e:
            raise MLModelError(f"Failed to save model: {e}") from e

    def load_model(self, filepath: str) -> None:
        """Load trained model from file.

        Args:
            filepath: Path to load model from

        Raises:
            MLModelError: If load fails
        """
        try:
            model_data = joblib.load(filepath)

            self._model = model_data['model']
            self._scaler = model_data['scaler']
            self._feature_weights = model_data.get('feature_weights', self._feature_weights)
            self._is_trained = model_data.get('is_trained', True)

            logger.info(f"Difficulty prediction model loaded from {filepath}")

        except Exception as e:
            raise MLModelError(f"Failed to load model: {e}") from e

    def is_trained(self) -> bool:
        """Check if model is trained.

        Returns:
            True if model is trained, False otherwise
        """
        return self._is_trained
