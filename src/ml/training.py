"""Advanced ML model training system for WORDLE solving - Phase 3 implementation.

This module provides comprehensive training capabilities including data generation,
model selection, hyperparameter optimization, and evaluation.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import SVR

from .. import MLModelError
from .features import FeatureExtractor
from .models import SimpleLinearModel, WordleMLModel
from .neural_models import NeuralWordleModel

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generates training data for ML models through game simulation."""

    def __init__(self, word_list: list[str], feature_extractor: FeatureExtractor) -> None:
        """Initialize training data generator.

        Args:
            word_list: List of valid words for training
            feature_extractor: Feature extractor for word analysis
        """
        self.word_list = [w.upper() for w in word_list]
        self.feature_extractor = feature_extractor

        logger.debug(f"TrainingDataGenerator initialized with {len(word_list)} words")

    def generate_guess_scoring_data(
        self,
        num_samples: int = 10000,
        strategies: list[str] | None = None,
        noise_level: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate training data for guess scoring models.

        Args:
            num_samples: Number of training samples to generate
            strategies: List of strategies to simulate
            noise_level: Amount of noise to add to targets

        Returns:
            Tuple of (features, targets) arrays
        """
        if strategies is None:
            strategies = ['entropy', 'frequency', 'position']

        logger.info(f"Generating {num_samples} guess scoring samples")

        features_list = []
        targets_list = []

        # Generate samples
        for _ in range(num_samples):
            # Random word and guess
            word = np.random.choice(self.word_list)
            guess = np.random.choice(self.word_list)

            # Random game state
            previous_guesses = []
            num_previous = np.random.randint(0, 4)

            for _ in range(num_previous):
                prev_guess = np.random.choice(self.word_list)
                prev_pattern = self._simulate_pattern(prev_guess, word)
                previous_guesses.append({'word': prev_guess, 'pattern': prev_pattern})

            # Extract features
            try:
                guess_features = self.feature_extractor.extract_features(
                    guess, self.word_list, previous_guesses
                )
                features_list.append(guess_features)

                # Calculate target score based on information gain simulation
                target_score = self._calculate_target_score(guess, word, previous_guesses)

                # Add some noise
                target_score = np.clip(target_score + np.random.normal(0, noise_level), 0, 1)
                targets_list.append(target_score)

            except Exception as e:
                logger.debug(f"Skipping sample due to feature extraction error: {e}")
                continue

        if not features_list:
            raise MLModelError("No valid training samples generated")

        features = np.array(features_list)
        targets = np.array(targets_list)

        logger.info(f"Generated {len(features)} valid training samples")
        return features, targets

    def generate_difficulty_prediction_data(
        self,
        num_samples: int = 5000
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate training data for word difficulty prediction.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Tuple of (features, targets) arrays
        """
        logger.info(f"Generating {num_samples} difficulty prediction samples")

        features_list = []
        targets_list = []

        selected_words = np.random.choice(self.word_list, num_samples, replace=True)

        for word in selected_words:
            try:
                # Extract word features
                word_features = self.feature_extractor.extract_features(word, [word], [])
                features_list.append(word_features)

                # Calculate difficulty based on multiple factors
                difficulty = self._calculate_word_difficulty(word)
                targets_list.append(difficulty)

            except Exception as e:
                logger.debug(f"Skipping word {word} due to feature extraction error: {e}")
                continue

        features = np.array(features_list)
        targets = np.array(targets_list)

        logger.info(f"Generated {len(features)} difficulty prediction samples")
        return features, targets

    def _simulate_pattern(self, guess: str, target: str) -> str:
        """Simulate WORDLE pattern for a guess.

        Args:
            guess: Guessed word
            target: Target word

        Returns:
            Pattern string (G/Y/X format)
        """
        pattern = []
        target_chars = list(target)

        # First pass: mark exact matches
        for i, (g_char, t_char) in enumerate(zip(guess, target, strict=False)):
            if g_char == t_char:
                pattern.append('G')
                target_chars[i] = None  # Mark as used
            else:
                pattern.append(None)  # To be determined

        # Second pass: mark yellow/gray
        for i, g_char in enumerate(guess):
            if pattern[i] is None:  # Not already green
                if g_char in target_chars:
                    pattern[i] = 'Y'
                    target_chars[target_chars.index(g_char)] = None  # Mark as used
                else:
                    pattern[i] = 'X'

        return ''.join(pattern)

    def _calculate_target_score(
        self,
        guess: str,
        target: str,
        previous_guesses: list[dict[str, str]]
    ) -> float:
        """Calculate target score for a guess based on information theory.

        Args:
            guess: The guess word
            target: The target word
            previous_guesses: Previous guesses and patterns

        Returns:
            Target score (0-1)
        """
        # Base score from letter frequency
        common_letters = set('ETAOINSHRDLU')
        frequency_score = sum(0.1 for c in guess if c in common_letters) / 5

        # Penalty for repeated letters
        unique_penalty = (5 - len(set(guess))) * 0.1

        # Bonus for vowel balance
        vowels = sum(1 for c in guess if c in 'AEIOU')
        balance_bonus = 0.2 if 2 <= vowels <= 3 else 0

        # Pattern information gain (simplified)
        pattern = self._simulate_pattern(guess, target)
        green_bonus = pattern.count('G') * 0.15
        yellow_bonus = pattern.count('Y') * 0.1

        # Avoid repetition of previous guesses
        repetition_penalty = 0.5 if guess in [pg['word'] for pg in previous_guesses] else 0

        score = frequency_score - unique_penalty + balance_bonus + green_bonus + yellow_bonus - repetition_penalty
        return np.clip(score, 0, 1)

    def _calculate_word_difficulty(self, word: str) -> float:
        """Calculate difficulty score for a word.

        Args:
            word: Word to calculate difficulty for

        Returns:
            Difficulty score (0-1)
        """
        difficulty = 0.0

        # Letter frequency penalty
        common_letters = set('ETAOINSHRDLU')
        uncommon_count = sum(1 for c in word if c not in common_letters)
        difficulty += uncommon_count * 0.15

        # Repeated letters increase difficulty
        unique_letters = len(set(word))
        if unique_letters < 5:
            difficulty += (5 - unique_letters) * 0.2

        # Vowel distribution
        vowels = sum(1 for c in word if c in 'AEIOU')
        if vowels < 2 or vowels > 3:
            difficulty += 0.15

        # Positional letter frequency (simplified)
        # Common starting letters
        if word[0] not in 'STCBMFPHDRLGWNK':
            difficulty += 0.1

        # Common ending letters
        if word[-1] not in 'SYEDTNALRKH':
            difficulty += 0.1

        return min(1.0, difficulty)


class ModelTrainer:
    """Advanced model trainer with hyperparameter optimization and evaluation."""

    def __init__(
        self,
        models_dir: str | None = None,
        results_dir: str | None = None
    ) -> None:
        """Initialize model trainer.

        Args:
            models_dir: Directory to save trained models
            results_dir: Directory to save training results
        """
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.results_dir = Path(results_dir) if results_dir else Path("results")

        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Available model types
        self.model_classes = {
            'linear': SimpleLinearModel,
            'neural_deep': lambda: NeuralWordleModel(architecture='deep'),
            'neural_cnn': lambda: NeuralWordleModel(architecture='cnn'),
            'random_forest': self._create_sklearn_wrapper(RandomForestRegressor),
            'gradient_boost': self._create_sklearn_wrapper(GradientBoostingRegressor),
            'svm': self._create_sklearn_wrapper(SVR)
        }

        self.training_results: dict[str, dict[str, Any]] = {}

        logger.debug(f"ModelTrainer initialized with {len(self.model_classes)} model types")

    def _create_sklearn_wrapper(self, sklearn_class):
        """Create a wrapper for sklearn models to match WordleMLModel interface."""

        class SklearnWrapper(WordleMLModel):
            def __init__(self, **kwargs):
                self.model = sklearn_class(**kwargs)
                self._is_trained = False

            def train(self, features: np.ndarray, targets: np.ndarray) -> None:
                self.model.fit(features, targets)
                self._is_trained = True

            def predict(self, features: np.ndarray) -> np.ndarray:
                if not self._is_trained:
                    raise MLModelError("Model must be trained before prediction")
                return self.model.predict(features)

            def save_model(self, filepath: str) -> None:
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(self.model, f)

            def load_model(self, filepath: str) -> None:
                import pickle
                with open(filepath, 'rb') as f:
                    self.model = pickle.load(f)
                self._is_trained = True

            def get_model_info(self) -> dict[str, Any]:
                return {
                    'model_type': sklearn_class.__name__,
                    'is_trained': self._is_trained
                }

        return SklearnWrapper

    def train_model(
        self,
        model_type: str,
        features: np.ndarray,
        targets: np.ndarray,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        model_params: dict[str, Any] | None = None,
        save_model: bool = True
    ) -> dict[str, Any]:
        """Train a single model with evaluation.

        Args:
            model_type: Type of model to train
            features: Training features
            targets: Training targets
            test_size: Fraction of data for testing
            validation_size: Fraction of remaining data for validation
            model_params: Model-specific parameters
            save_model: Whether to save the trained model

        Returns:
            Dictionary with training results
        """
        if model_type not in self.model_classes:
            raise MLModelError(f"Unknown model type: {model_type}")

        logger.info(f"Training {model_type} model with {len(features)} samples")

        start_time = time.time()

        # Split data
        x_temp, x_test, y_temp, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=42
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp, test_size=validation_size, random_state=42
        )

        # Create model
        if model_params is None:
            model_params = {}

        try:
            if model_type.startswith('neural'):
                model = self.model_classes[model_type]()

                # Neural models have different training interface
                if hasattr(model, 'train'):
                    model.train(
                        x_train, y_train,
                        validation_split=0.0,  # We already split the data
                        **model_params
                    )
            else:
                model = self.model_classes[model_type](**model_params)
                model.train(x_train, y_train)

            # Evaluate model
            train_pred = model.predict(x_train)
            val_pred = model.predict(x_val)
            test_pred = model.predict(x_test)

            # Calculate metrics
            results = {
                'model_type': model_type,
                'training_time': time.time() - start_time,
                'data_splits': {
                    'train_size': len(x_train),
                    'val_size': len(x_val),
                    'test_size': len(x_test)
                },
                'train_metrics': self._calculate_metrics(y_train, train_pred),
                'val_metrics': self._calculate_metrics(y_val, val_pred),
                'test_metrics': self._calculate_metrics(y_test, test_pred),
                'model_info': model.get_model_info(),
                'timestamp': datetime.now().isoformat()
            }

            # Cross-validation (for non-neural models)
            if not model_type.startswith('neural'):
                try:
                    cv_scores = cross_val_score(
                        model.model if hasattr(model, 'model') else model,
                        x_temp, y_temp, cv=5, scoring='neg_mean_squared_error'
                    )
                    results['cv_metrics'] = {
                        'mean_mse': -cv_scores.mean(),
                        'std_mse': cv_scores.std(),
                        'scores': (-cv_scores).tolist()
                    }
                except Exception as e:
                    logger.warning(f"Cross-validation failed for {model_type}: {e}")

            # Save model
            if save_model:
                model_path = self.models_dir / f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                model.save_model(str(model_path))
                results['model_path'] = str(model_path)

            # Store results
            self.training_results[model_type] = results

            logger.info(f"{model_type} training completed in {results['training_time']:.2f}s")
            logger.info(f"Test MSE: {results['test_metrics']['mse']:.6f}, R²: {results['test_metrics']['r2']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Training failed for {model_type}: {e}")
            raise MLModelError(f"Training failed for {model_type}: {e}") from e

    def train_multiple_models(
        self,
        model_types: list[str],
        features: np.ndarray,
        targets: np.ndarray,
        model_params_dict: dict[str, dict[str, Any]] | None = None,
        parallel: bool = True
    ) -> dict[str, dict[str, Any]]:
        """Train multiple models in parallel.

        Args:
            model_types: List of model types to train
            features: Training features
            targets: Training targets
            model_params_dict: Parameters for each model type
            parallel: Whether to train models in parallel

        Returns:
            Dictionary with results for each model
        """
        logger.info(f"Training {len(model_types)} models: {model_types}")

        if model_params_dict is None:
            model_params_dict = {}

        results = {}

        if parallel and len(model_types) > 1:
            # Parallel training
            with ThreadPoolExecutor(max_workers=min(4, len(model_types))) as executor:
                future_to_model = {
                    executor.submit(
                        self.train_model,
                        model_type,
                        features,
                        targets,
                        model_params=model_params_dict.get(model_type, {})
                    ): model_type
                    for model_type in model_types
                }

                for future in as_completed(future_to_model):
                    model_type = future_to_model[future]
                    try:
                        result = future.result()
                        results[model_type] = result
                    except Exception as e:
                        logger.error(f"Parallel training failed for {model_type}: {e}")
                        results[model_type] = {'error': str(e)}
        else:
            # Sequential training
            for model_type in model_types:
                try:
                    result = self.train_model(
                        model_type,
                        features,
                        targets,
                        model_params=model_params_dict.get(model_type, {})
                    )
                    results[model_type] = result
                except Exception as e:
                    logger.error(f"Sequential training failed for {model_type}: {e}")
                    results[model_type] = {'error': str(e)}

        return results

    def hyperparameter_optimization(
        self,
        model_type: str,
        features: np.ndarray,
        targets: np.ndarray,
        param_grid: dict[str, list[Any]],
        cv_folds: int = 5
    ) -> dict[str, Any]:
        """Perform hyperparameter optimization using grid search.

        Args:
            model_type: Type of model to optimize
            features: Training features
            targets: Training targets
            param_grid: Parameter grid for optimization
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with optimization results
        """
        if model_type not in self.model_classes or model_type.startswith('neural'):
            raise MLModelError(f"Hyperparameter optimization not supported for {model_type}")

        logger.info(f"Starting hyperparameter optimization for {model_type}")

        try:
            # Create base model
            base_model = self.model_classes[model_type]()

            # Grid search
            grid_search = GridSearchCV(
                base_model.model if hasattr(base_model, 'model') else base_model,
                param_grid,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(features, targets)

            results = {
                'model_type': model_type,
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_,
                'cv_results': {
                    'mean_test_scores': (-grid_search.cv_results_['mean_test_score']).tolist(),
                    'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                },
                'n_splits': cv_folds,
                'param_grid': param_grid
            }

            logger.info(f"Optimization completed. Best params: {grid_search.best_params_}")
            logger.info(f"Best CV score: {-grid_search.best_score_:.6f}")

            return results

        except Exception as e:
            logger.error(f"Hyperparameter optimization failed for {model_type}: {e}")
            raise MLModelError(f"Hyperparameter optimization failed: {e}") from e

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'max_error': np.max(np.abs(y_true - y_pred)),
            'mean_prediction': np.mean(y_pred),
            'std_prediction': np.std(y_pred)
        }

    def generate_training_report(
        self,
        output_path: str | None = None,
        include_plots: bool = True
    ) -> str:
        """Generate comprehensive training report.

        Args:
            output_path: Path to save report
            include_plots: Whether to include plots

        Returns:
            Path to generated report
        """
        if not self.training_results:
            raise MLModelError("No training results available")

        # Generate report
        report_data = {
            'summary': {
                'num_models': len(self.training_results),
                'best_model': self._find_best_model(),
                'generation_time': datetime.now().isoformat()
            },
            'detailed_results': self.training_results
        }

        # Save report
        if output_path is None:
            output_path = self.results_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        # Generate plots if requested
        if include_plots:
            self._generate_training_plots(str(output_path).replace('.json', '_plots'))

        logger.info(f"Training report generated: {output_path}")
        return str(output_path)

    def _find_best_model(self) -> dict[str, Any]:
        """Find the best performing model.

        Returns:
            Dictionary with best model information
        """
        best_model = None
        best_score = float('inf')

        for model_type, results in self.training_results.items():
            if 'error' in results:
                continue

            test_mse = results.get('test_metrics', {}).get('mse', float('inf'))
            if test_mse < best_score:
                best_score = test_mse
                best_model = {
                    'model_type': model_type,
                    'test_mse': test_mse,
                    'test_r2': results.get('test_metrics', {}).get('r2', 0),
                    'training_time': results.get('training_time', 0)
                }

        return best_model or {'model_type': 'none', 'test_mse': float('inf')}

    def _generate_training_plots(self, output_dir: str) -> None:
        """Generate training visualization plots.

        Args:
            output_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt

            plot_dir = Path(output_dir)
            plot_dir.mkdir(exist_ok=True)

            # Set style
            plt.style.use('seaborn-v0_8')

            # Model comparison plot
            self._plot_model_comparison(plot_dir)

            # Training metrics plot
            self._plot_training_metrics(plot_dir)

            logger.info(f"Training plots saved to {plot_dir}")

        except Exception as e:
            logger.warning(f"Failed to generate training plots: {e}")

    def _plot_model_comparison(self, plot_dir: Path) -> None:
        """Generate model comparison plot."""
        # Extract data for plotting
        models = []
        test_mse = []
        test_r2 = []
        training_time = []

        for model_type, results in self.training_results.items():
            if 'error' not in results:
                models.append(model_type)
                test_mse.append(results.get('test_metrics', {}).get('mse', 0))
                test_r2.append(results.get('test_metrics', {}).get('r2', 0))
                training_time.append(results.get('training_time', 0))

        if not models:
            return

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # MSE comparison
        axes[0].bar(models, test_mse)
        axes[0].set_title('Test MSE by Model')
        axes[0].set_ylabel('Mean Squared Error')
        axes[0].tick_params(axis='x', rotation=45)

        # R² comparison
        axes[1].bar(models, test_r2)
        axes[1].set_title('Test R² by Model')
        axes[1].set_ylabel('R² Score')
        axes[1].tick_params(axis='x', rotation=45)

        # Training time comparison
        axes[2].bar(models, training_time)
        axes[2].set_title('Training Time by Model')
        axes[2].set_ylabel('Time (seconds)')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(plot_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_training_metrics(self, plot_dir: Path) -> None:
        """Generate training metrics plots."""
        # Create metrics comparison dataframe
        metrics_data = []

        for model_type, results in self.training_results.items():
            if 'error' not in results:
                for split in ['train', 'val', 'test']:
                    metrics = results.get(f'{split}_metrics', {})
                    metrics_data.append({
                        'model': model_type,
                        'split': split,
                        'mse': metrics.get('mse', 0),
                        'r2': metrics.get('r2', 0),
                        'mae': metrics.get('mae', 0)
                    })

        if not metrics_data:
            return

        df = pd.DataFrame(metrics_data)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # MSE by split
        sns.barplot(data=df, x='model', y='mse', hue='split', ax=axes[0, 0])
        axes[0, 0].set_title('MSE by Model and Split')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # R² by split
        sns.barplot(data=df, x='model', y='r2', hue='split', ax=axes[0, 1])
        axes[0, 1].set_title('R² by Model and Split')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # MAE by split
        sns.barplot(data=df, x='model', y='mae', hue='split', ax=axes[1, 0])
        axes[1, 0].set_title('MAE by Model and Split')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Overfitting analysis (train vs test R²)
        train_r2 = df[df['split'] == 'train'].set_index('model')['r2']
        test_r2 = df[df['split'] == 'test'].set_index('model')['r2']

        axes[1, 1].scatter(train_r2, test_r2)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1, 1].set_xlabel('Train R²')
        axes[1, 1].set_ylabel('Test R²')
        axes[1, 1].set_title('Overfitting Analysis')

        # Add model labels
        for model in train_r2.index:
            axes[1, 1].annotate(model, (train_r2[model], test_r2[model]),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.tight_layout()
        plt.savefig(plot_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()


class MLTrainingPipeline:
    """Complete ML training pipeline orchestrator."""

    def __init__(
        self,
        word_list: list[str],
        models_dir: str = "models",
        results_dir: str = "results",
        data_dir: str = "training_data"
    ) -> None:
        """Initialize training pipeline.

        Args:
            word_list: List of words for training
            models_dir: Directory for saving models
            results_dir: Directory for saving results
            data_dir: Directory for saving training data
        """
        self.word_list = word_list
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)

        # Create directories
        for directory in [self.models_dir, self.results_dir, self.data_dir]:
            directory.mkdir(exist_ok=True)

        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.data_generator = TrainingDataGenerator(word_list, self.feature_extractor)
        self.model_trainer = ModelTrainer(str(self.models_dir), str(self.results_dir))

        logger.info("MLTrainingPipeline initialized")

    def run_full_training_pipeline(
        self,
        num_guess_samples: int = 50000,
        num_difficulty_samples: int = 10000,
        model_types: list[str] | None = None,
        hyperparameter_optimize: bool = True,
        generate_report: bool = True
    ) -> dict[str, Any]:
        """Run the complete training pipeline.

        Args:
            num_guess_samples: Number of guess scoring samples
            num_difficulty_samples: Number of difficulty prediction samples
            model_types: List of model types to train
            hyperparameter_optimize: Whether to optimize hyperparameters
            generate_report: Whether to generate training report

        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting full ML training pipeline")
        pipeline_start = time.time()

        if model_types is None:
            model_types = ['linear', 'neural_deep', 'random_forest', 'gradient_boost']

        results = {
            'pipeline_start': datetime.now().isoformat(),
            'configuration': {
                'num_guess_samples': num_guess_samples,
                'num_difficulty_samples': num_difficulty_samples,
                'model_types': model_types,
                'hyperparameter_optimize': hyperparameter_optimize
            }
        }

        try:
            # Step 1: Generate training data
            logger.info("Step 1: Generating training data")

            guess_features, guess_targets = self.data_generator.generate_guess_scoring_data(
                num_guess_samples
            )

            difficulty_features, difficulty_targets = self.data_generator.generate_difficulty_prediction_data(
                num_difficulty_samples
            )

            # Save training data
            np.savez(
                self.data_dir / f"guess_scoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz",
                features=guess_features,
                targets=guess_targets
            )

            np.savez(
                self.data_dir / f"difficulty_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz",
                features=difficulty_features,
                targets=difficulty_targets
            )

            results['data_generation'] = {
                'guess_samples': len(guess_features),
                'difficulty_samples': len(difficulty_features),
                'guess_features_dim': guess_features.shape[1],
                'difficulty_features_dim': difficulty_features.shape[1]
            }

            # Step 2: Train models
            logger.info("Step 2: Training models")

            # Focus on guess scoring models (more important for WORDLE)
            training_results = self.model_trainer.train_multiple_models(
                model_types,
                guess_features,
                guess_targets
            )

            results['model_training'] = training_results

            # Step 3: Hyperparameter optimization (for selected models)
            if hyperparameter_optimize:
                logger.info("Step 3: Hyperparameter optimization")

                optimization_results = {}

                # Define parameter grids
                param_grids = {
                    'random_forest': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10]
                    },
                    'gradient_boost': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.1, 0.05, 0.01],
                        'max_depth': [5, 10, 15]
                    }
                }

                for model_type in ['random_forest', 'gradient_boost']:
                    if model_type in model_types:
                        try:
                            opt_result = self.model_trainer.hyperparameter_optimization(
                                model_type,
                                guess_features,
                                guess_targets,
                                param_grids[model_type]
                            )
                            optimization_results[model_type] = opt_result
                        except Exception as e:
                            logger.warning(f"Optimization failed for {model_type}: {e}")

                results['hyperparameter_optimization'] = optimization_results

            # Step 4: Generate report
            if generate_report:
                logger.info("Step 4: Generating training report")

                report_path = self.model_trainer.generate_training_report(include_plots=True)
                results['report_path'] = report_path

            # Pipeline summary
            pipeline_time = time.time() - pipeline_start
            results['pipeline_summary'] = {
                'total_time': pipeline_time,
                'completion_time': datetime.now().isoformat(),
                'models_trained': len([r for r in training_results.values() if 'error' not in r]),
                'best_model': self.model_trainer._find_best_model(),
                'success': True
            }

            logger.info(f"Training pipeline completed successfully in {pipeline_time:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            results['pipeline_summary'] = {
                'total_time': time.time() - pipeline_start,
                'completion_time': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            }
            raise MLModelError(f"Training pipeline failed: {e}") from e
