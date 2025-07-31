"""Main WORDLE solving engine.

This module provides the core WordleSolver class that orchestrates the solving
process by combining entropy calculations, pattern matching, and ML predictions.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data.patterns import PatternMatcher
from ..data.words import WordListManager
from ..ml.prediction import PredictionEngine
from ..utils.helpers import validate_word
from .entropy import EntropyCalculator
from .strategy import SolvingStrategy

logger = logging.getLogger(__name__)


@dataclass
class GuessResult:
    """Result of a single guess in the game.

    Attributes:
        guess: The word that was guessed
        pattern: Feedback pattern (G=green, Y=yellow, X=gray)
        remaining_words: Number of remaining possible words
        entropy: Information entropy of the guess
        ml_score: Machine learning confidence score
        processing_time: Time taken to calculate the guess
    """
    guess: str
    pattern: str
    remaining_words: int
    entropy: float
    ml_score: float
    processing_time: float


@dataclass
class GameResult:
    """Result of a complete game.

    Attributes:
        target_word: The target word (if known)
        guesses: List of all guesses made
        solved: Whether the puzzle was solved
        attempts: Number of attempts used
        total_time: Total solving time
        strategy_used: Strategy that was used
    """
    target_word: str | None
    guesses: list[GuessResult]
    solved: bool
    attempts: int
    total_time: float
    strategy_used: str


@dataclass
class BenchmarkResult:
    """Results from benchmark testing.

    Attributes:
        total_words: Total number of words tested
        solved_count: Number of words solved
        average_attempts: Average number of attempts
        success_rate: Percentage of words solved
        total_time: Total benchmark time
        strategy_performance: Performance by strategy
    """
    total_words: int
    solved_count: int
    average_attempts: float
    success_rate: float
    total_time: float
    strategy_performance: dict[str, dict[str, float]]


class WordleSolver:
    """Main WORDLE solving engine.

    This class orchestrates the solving process by combining multiple strategies
    including entropy calculations, pattern matching, and machine learning predictions.
    """

    def __init__(
        self,
        strategy: str = "hybrid",
        word_list_path: Path | None = None,
        max_attempts: int = 6
    ) -> None:
        """Initialize the WORDLE solver.

        Args:
            strategy: Solving strategy ("entropy", "ml", "hybrid")
            word_list_path: Optional path to custom word list
            max_attempts: Maximum number of guesses allowed

        Raises:
            WordleAIException: If initialization fails
        """
        logger.info(f"Initializing WordleSolver with strategy: {strategy}")

        self.strategy_name = strategy
        self.max_attempts = max_attempts

        try:
            # Initialize core components
            self._word_manager = WordListManager(word_list_path)
            self._pattern_matcher = PatternMatcher()
            self._entropy_calculator = EntropyCalculator()
            self._prediction_engine = PredictionEngine()

            # Initialize solving strategy
            self._strategy = SolvingStrategy(
                strategy_type=strategy,
                entropy_calculator=self._entropy_calculator,
                prediction_engine=self._prediction_engine
            )

            logger.info("WordleSolver initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize WordleSolver: {e}")
            raise

    def solve_interactive(self, display) -> GameResult:
        """Solve WORDLE puzzle interactively with user input.

        Args:
            display: Display interface for user interaction

        Returns:
            Game result with solving statistics

        Raises:
            WordleAIException: If solving fails
        """
        logger.info("Starting interactive solving session")
        start_time = time.time()

        guesses: list[GuessResult] = []
        possible_words = self._word_manager.get_valid_words()

        try:
            for attempt in range(1, self.max_attempts + 1):
                logger.debug(f"Attempt {attempt}/{self.max_attempts}")

                # Get next guess recommendation
                guess_start = time.time()
                guess = self._strategy.get_best_guess(possible_words, guesses)
                processing_time = time.time() - guess_start

                logger.info(f"Recommended guess: {guess} (took {processing_time:.3f}s)")

                # Display guess and get user feedback
                display.show_guess_recommendation(guess, len(possible_words))
                pattern = display.get_pattern_feedback(guess)

                # Calculate guess statistics
                entropy = self._entropy_calculator.calculate_guess_entropy(
                    guess, possible_words
                )
                ml_score = self._prediction_engine.score_guess(
                    guess, possible_words, guesses
                )

                # Record guess result
                guess_result = GuessResult(
                    guess=guess,
                    pattern=pattern,
                    remaining_words=len(possible_words),
                    entropy=entropy,
                    ml_score=ml_score,
                    processing_time=processing_time
                )
                guesses.append(guess_result)

                # Check if solved
                if pattern == "GGGGG":
                    logger.info(f"Puzzle solved in {attempt} attempts!")
                    break

                # Filter words based on feedback
                possible_words = self._pattern_matcher.filter_words(
                    possible_words, guess, pattern
                )

                logger.debug(f"Remaining words after filtering: {len(possible_words)}")

                if not possible_words:
                    logger.warning("No remaining possible words - invalid feedback?")
                    display.show_error("No valid words remaining. Please check feedback.")
                    break

            total_time = time.time() - start_time
            solved = len(guesses) > 0 and guesses[-1].pattern == "GGGGG"

            result = GameResult(
                target_word=None,
                guesses=guesses,
                solved=solved,
                attempts=len(guesses),
                total_time=total_time,
                strategy_used=self.strategy_name
            )

            logger.info(f"Interactive session completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Interactive solving failed: {e}", exc_info=True)
            raise

    def solve_word(self, target_word: str) -> GameResult:
        """Solve for a specific target word.

        Args:
            target_word: The word to solve for

        Returns:
            Game result with solving statistics

        Raises:
            WordleAIException: If word is invalid or solving fails
        """
        target_word = target_word.upper()
        validate_word(target_word)

        logger.info(f"Solving for target word: {target_word}")
        start_time = time.time()

        guesses: list[GuessResult] = []
        possible_words = self._word_manager.get_valid_words()

        try:
            for attempt in range(1, self.max_attempts + 1):
                logger.debug(f"Attempt {attempt}/{self.max_attempts}")

                # Get next guess
                guess_start = time.time()
                guess = self._strategy.get_best_guess(possible_words, guesses)
                processing_time = time.time() - guess_start

                # Generate pattern feedback
                pattern = self._pattern_matcher.generate_pattern(guess, target_word)

                # Calculate statistics
                entropy = self._entropy_calculator.calculate_guess_entropy(
                    guess, possible_words
                )
                ml_score = self._prediction_engine.score_guess(
                    guess, possible_words, guesses
                )

                # Record guess
                guess_result = GuessResult(
                    guess=guess,
                    pattern=pattern,
                    remaining_words=len(possible_words),
                    entropy=entropy,
                    ml_score=ml_score,
                    processing_time=processing_time
                )
                guesses.append(guess_result)

                logger.info(f"Guess {attempt}: {guess} -> {pattern}")

                # Check if solved
                if pattern == "GGGGG":
                    logger.info(f"Solved '{target_word}' in {attempt} attempts!")
                    break

                # Filter remaining words
                possible_words = self._pattern_matcher.filter_words(
                    possible_words, guess, pattern
                )

                if not possible_words:
                    logger.error(f"No remaining words for target '{target_word}'")
                    break

            total_time = time.time() - start_time
            solved = len(guesses) > 0 and guesses[-1].pattern == "GGGGG"

            result = GameResult(
                target_word=target_word,
                guesses=guesses,
                solved=solved,
                attempts=len(guesses),
                total_time=total_time,
                strategy_used=self.strategy_name
            )

            return result

        except Exception as e:
            logger.error(f"Failed to solve word '{target_word}': {e}", exc_info=True)
            raise

    def run_benchmark(self, word_count: int | None = None) -> BenchmarkResult:
        """Run benchmark testing on word list.

        Args:
            word_count: Number of words to test (None for all)

        Returns:
            Benchmark results

        Raises:
            WordleAIException: If benchmark fails
        """
        logger.info("Starting benchmark run")
        start_time = time.time()

        test_words = self._word_manager.get_answer_words()
        if word_count:
            test_words = test_words[:word_count]

        logger.info(f"Benchmarking {len(test_words)} words")

        results: list[GameResult] = []

        try:
            for i, word in enumerate(test_words, 1):
                if i % 100 == 0:
                    logger.info(f"Benchmark progress: {i}/{len(test_words)}")

                result = self.solve_word(word)
                results.append(result)

            # Calculate statistics
            solved_count = sum(1 for r in results if r.solved)
            total_attempts = sum(r.attempts for r in results)

            benchmark_result = BenchmarkResult(
                total_words=len(test_words),
                solved_count=solved_count,
                average_attempts=total_attempts / len(results) if results else 0,
                success_rate=(solved_count / len(results)) * 100 if results else 0,
                total_time=time.time() - start_time,
                strategy_performance=self._analyze_strategy_performance(results)
            )

            logger.info(f"Benchmark completed: {benchmark_result}")
            return benchmark_result

        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            raise

    def train_models(self) -> None:
        """Train machine learning models on historical data.

        Raises:
            MLModelError: If training fails
        """
        logger.info("Starting ML model training")

        try:
            # Generate training data from benchmark runs
            training_data = self._generate_training_data()

            # Train prediction models
            self._prediction_engine.train_models(training_data)

            logger.info("ML model training completed successfully")

        except Exception as e:
            logger.error(f"ML model training failed: {e}", exc_info=True)
            raise

    def _analyze_strategy_performance(self, results: list[GameResult]) -> dict[str, dict[str, float]]:
        """Analyze performance by strategy.

        Args:
            results: List of game results

        Returns:
            Performance metrics by strategy
        """
        performance = {
            self.strategy_name: {
                "success_rate": (sum(1 for r in results if r.solved) / len(results)) * 100,
                "avg_attempts": sum(r.attempts for r in results) / len(results),
                "avg_time": sum(r.total_time for r in results) / len(results)
            }
        }
        return performance

    def _generate_training_data(self) -> list[dict[str, Any]]:
        """Generate training data for ML models.

        Returns:
            Training data samples
        """
        logger.info("Generating training data")

        # For now, return empty list - will be implemented in Phase 3
        return []
