"""Statistical analysis tools for WORDLE AI performance.

This module provides comprehensive statistical analysis of solving performance,
word patterns, and game outcomes.
"""

import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import WordleAIException

logger = logging.getLogger(__name__)


@dataclass
class GameStatistics:
    """Container for game statistics."""
    word: str
    attempts: int
    success: bool
    strategy_used: str
    time_taken: float
    guesses: list[str]
    entropies: list[float]
    ml_scores: list[float]


@dataclass
class AnalysisResult:
    """Container for statistical analysis results."""
    total_games: int
    success_rate: float
    average_attempts: float
    median_attempts: float
    std_attempts: float
    min_attempts: int
    max_attempts: int
    attempt_distribution: dict[int, int]
    strategy_performance: dict[str, dict[str, float]]
    word_difficulty_stats: dict[str, float]
    letter_frequency_analysis: dict[str, float]
    position_analysis: dict[int, dict[str, float]]


class StatisticalAnalyzer:
    """Advanced statistical analysis for WORDLE solving performance.

    Provides comprehensive statistical tools to analyze game outcomes,
    strategy effectiveness, and word patterns.
    """

    def __init__(self) -> None:
        """Initialize statistical analyzer."""
        logger.debug("Initializing StatisticalAnalyzer")

        self._game_history: list[GameStatistics] = []
        self._analysis_cache: dict[str, Any] = {}

        logger.debug("StatisticalAnalyzer initialized")

    def add_game_result(
        self,
        word: str,
        attempts: int,
        success: bool,
        strategy_used: str,
        time_taken: float,
        guesses: list[str],
        entropies: list[float] | None = None,
        ml_scores: list[float] | None = None
    ) -> None:
        """Add a game result to the analysis dataset.

        Args:
            word: Target word that was being solved
            attempts: Number of attempts taken
            success: Whether the puzzle was solved successfully
            strategy_used: Strategy that was used (entropy/ml/hybrid)
            time_taken: Time taken to solve in seconds
            guesses: List of guesses made
            entropies: List of entropy values for each guess
            ml_scores: List of ML scores for each guess
        """
        game_stats = GameStatistics(
            word=word,
            attempts=attempts,
            success=success,
            strategy_used=strategy_used,
            time_taken=time_taken,
            guesses=guesses,
            entropies=entropies or [],
            ml_scores=ml_scores or []
        )

        self._game_history.append(game_stats)
        self._analysis_cache.clear()  # Clear cache when new data is added

        logger.debug(f"Added game result: {word} in {attempts} attempts ({strategy_used})")

    def analyze_performance(self) -> AnalysisResult:
        """Perform comprehensive statistical analysis of game performance.

        Returns:
            AnalysisResult containing detailed statistical analysis

        Raises:
            WordleAIException: If no game data is available
        """
        if not self._game_history:
            raise WordleAIException("No game data available for analysis")

        cache_key = f"performance_analysis_{len(self._game_history)}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        logger.info(f"Analyzing performance for {len(self._game_history)} games")

        # Basic statistics
        successful_games = [g for g in self._game_history if g.success]
        attempts_list = [g.attempts for g in successful_games]

        if not attempts_list:
            raise WordleAIException("No successful games to analyze")

        # Calculate basic metrics
        total_games = len(self._game_history)
        success_rate = len(successful_games) / total_games
        average_attempts = statistics.mean(attempts_list)
        median_attempts = statistics.median(attempts_list)
        std_attempts = statistics.stdev(attempts_list) if len(attempts_list) > 1 else 0.0
        min_attempts = min(attempts_list)
        max_attempts = max(attempts_list)

        # Attempt distribution
        attempt_distribution = Counter(attempts_list)

        # Strategy performance analysis
        strategy_performance = self._analyze_strategy_performance()

        # Word difficulty analysis
        word_difficulty_stats = self._analyze_word_difficulty()

        # Letter frequency analysis
        letter_frequency_analysis = self._analyze_letter_frequencies()

        # Position analysis
        position_analysis = self._analyze_position_patterns()

        result = AnalysisResult(
            total_games=total_games,
            success_rate=success_rate,
            average_attempts=average_attempts,
            median_attempts=median_attempts,
            std_attempts=std_attempts,
            min_attempts=min_attempts,
            max_attempts=max_attempts,
            attempt_distribution=dict(attempt_distribution),
            strategy_performance=strategy_performance,
            word_difficulty_stats=word_difficulty_stats,
            letter_frequency_analysis=letter_frequency_analysis,
            position_analysis=position_analysis
        )

        self._analysis_cache[cache_key] = result
        logger.info("Performance analysis completed")

        return result

    def _analyze_strategy_performance(self) -> dict[str, dict[str, float]]:
        """Analyze performance by strategy type.

        Returns:
            Dictionary mapping strategy names to performance metrics
        """
        strategy_stats = defaultdict(list)

        for game in self._game_history:
            if game.success:
                strategy_stats[game.strategy_used].append(game.attempts)

        performance = {}
        for strategy, attempts_list in strategy_stats.items():
            if attempts_list:
                performance[strategy] = {
                    "average_attempts": statistics.mean(attempts_list),
                    "median_attempts": statistics.median(attempts_list),
                    "success_rate": len(attempts_list) / len([g for g in self._game_history if g.strategy_used == strategy]),
                    "std_attempts": statistics.stdev(attempts_list) if len(attempts_list) > 1 else 0.0,
                    "games_played": len([g for g in self._game_history if g.strategy_used == strategy])
                }

        return performance

    def _analyze_word_difficulty(self) -> dict[str, float]:
        """Analyze word difficulty based on solving attempts.

        Returns:
            Dictionary mapping difficulty categories to average attempts
        """
        word_attempts = {}
        for game in self._game_history:
            if game.success:
                word_attempts[game.word] = game.attempts

        if not word_attempts:
            return {}

        # Calculate difficulty categories
        attempts_values = list(word_attempts.values())
        q1 = np.percentile(attempts_values, 25)
        q2 = np.percentile(attempts_values, 50)
        q3 = np.percentile(attempts_values, 75)

        difficulty_stats = {
            "easy_threshold": q1,
            "medium_threshold": q2,
            "hard_threshold": q3,
            "average_difficulty": statistics.mean(attempts_values),
            "difficulty_variance": statistics.variance(attempts_values) if len(attempts_values) > 1 else 0.0
        }

        return difficulty_stats

    def _analyze_letter_frequencies(self) -> dict[str, float]:
        """Analyze letter frequency patterns in target words.

        Returns:
            Dictionary mapping letters to their frequency in target words
        """
        letter_counts = Counter()
        total_letters = 0

        for game in self._game_history:
            for letter in game.word:
                letter_counts[letter] += 1
                total_letters += 1

        if total_letters == 0:
            return {}

        return {letter: count / total_letters for letter, count in letter_counts.items()}

    def _analyze_position_patterns(self) -> dict[int, dict[str, float]]:
        """Analyze letter frequency patterns by position.

        Returns:
            Dictionary mapping positions to letter frequency dictionaries
        """
        position_stats = defaultdict(lambda: defaultdict(int))
        position_totals = defaultdict(int)

        for game in self._game_history:
            for i, letter in enumerate(game.word):
                position_stats[i][letter] += 1
                position_totals[i] += 1

        # Convert to frequencies
        result = {}
        for pos in range(5):  # WORDLE words are 5 letters
            if position_totals[pos] > 0:
                result[pos] = {
                    letter: count / position_totals[pos]
                    for letter, count in position_stats[pos].items()
                }

        return result

    def generate_performance_report(self, analysis: AnalysisResult) -> str:
        """Generate a comprehensive performance report.

        Args:
            analysis: Analysis result from analyze_performance()

        Returns:
            Formatted performance report string
        """
        report_lines = [
            "📊 WORDLE AI PERFORMANCE ANALYSIS",
            "=" * 50,
            "",
            f"Total Games Analyzed: {analysis.total_games:,}",
            f"Success Rate: {analysis.success_rate:.1%}",
            f"Average Attempts: {analysis.average_attempts:.2f}",
            f"Median Attempts: {analysis.median_attempts:.1f}",
            f"Standard Deviation: {analysis.std_attempts:.2f}",
            f"Range: {analysis.min_attempts} - {analysis.max_attempts} attempts",
            "",
            "📈 ATTEMPT DISTRIBUTION:",
        ]

        # Add attempt distribution
        for attempts in sorted(analysis.attempt_distribution.keys()):
            count = analysis.attempt_distribution[attempts]
            percentage = count / analysis.total_games * 100
            bar = "█" * int(percentage / 2)
            report_lines.append(f"  {attempts} attempts: {count:,} games ({percentage:.1f}%) {bar}")

        # Add strategy performance
        if analysis.strategy_performance:
            report_lines.extend([
                "",
                "🎯 STRATEGY PERFORMANCE:",
            ])

            for strategy, stats in analysis.strategy_performance.items():
                report_lines.extend([
                    f"  {strategy.upper()}:",
                    f"    Average: {stats['average_attempts']:.2f} attempts",
                    f"    Success Rate: {stats['success_rate']:.1%}",
                    f"    Games: {stats['games_played']:,}",
                ])

        # Add difficulty analysis
        if analysis.word_difficulty_stats:
            report_lines.extend([
                "",
                "🎲 WORD DIFFICULTY ANALYSIS:",
                f"  Average Difficulty: {analysis.word_difficulty_stats.get('average_difficulty', 0):.2f} attempts",
                f"  Difficulty Variance: {analysis.word_difficulty_stats.get('difficulty_variance', 0):.2f}",
            ])

        return "\n".join(report_lines)

    def export_analysis_data(self, filepath: str, analysis: AnalysisResult) -> None:
        """Export analysis data to CSV file.

        Args:
            filepath: Path to save the CSV file
            analysis: Analysis result to export

        Raises:
            WordleAIException: If export fails
        """
        try:
            import pandas as pd

            # Prepare data for export
            game_data = []
            for game in self._game_history:
                game_data.append({
                    "word": game.word,
                    "attempts": game.attempts,
                    "success": game.success,
                    "strategy": game.strategy_used,
                    "time_taken": game.time_taken,
                    "guesses": ",".join(game.guesses)
                })

            df = pd.DataFrame(game_data)
            df.to_csv(filepath, index=False)

            logger.info(f"Analysis data exported to {filepath}")

        except ImportError:
            raise WordleAIException("pandas is required for data export") from None
        except Exception as e:
            raise WordleAIException(f"Failed to export analysis data: {e}") from e

    def clear_history(self) -> None:
        """Clear all game history and analysis cache."""
        self._game_history.clear()
        self._analysis_cache.clear()
        logger.info("Game history and analysis cache cleared")

    def get_game_count(self) -> int:
        """Get the number of games in the analysis dataset.

        Returns:
            Number of games analyzed
        """
        return len(self._game_history)
