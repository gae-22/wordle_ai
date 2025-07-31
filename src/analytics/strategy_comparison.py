"""Strategy comparison tools for WORDLE AI.

This module provides tools to compare different solving strategies
and analyze their relative performance across various scenarios.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .. import WordleAIException
from ..solver.engine import WordleSolver

logger = logging.getLogger(__name__)


class ComparisonMetric(Enum):
    """Available metrics for strategy comparison."""
    SUCCESS_RATE = "success_rate"
    AVERAGE_ATTEMPTS = "average_attempts"
    MEDIAN_ATTEMPTS = "median_attempts"
    TIME_EFFICIENCY = "time_efficiency"
    WORST_CASE = "worst_case"
    CONSISTENCY = "consistency"


@dataclass
class StrategyResult:
    """Results for a single strategy."""
    strategy_name: str
    success_rate: float
    average_attempts: float
    median_attempts: float
    std_attempts: float
    total_time: float
    average_time: float
    worst_case_attempts: int
    games_played: int
    failed_words: list[str]
    attempt_distribution: dict[int, int]


@dataclass
class ComparisonResult:
    """Results of strategy comparison."""
    strategies: dict[str, StrategyResult]
    winner_by_metric: dict[str, str]
    statistical_significance: dict[tuple[str, str], dict[str, float]]
    recommendations: list[str]


class StrategyComparator:
    """Advanced strategy comparison and analysis tool.

    Provides comprehensive comparison of different solving strategies
    across multiple metrics and scenarios.
    """

    def __init__(self) -> None:
        """Initialize strategy comparator."""
        logger.debug("Initializing StrategyComparator")

        self._comparison_cache: dict[str, ComparisonResult] = {}

        logger.debug("StrategyComparator initialized")

    def compare_strategies(
        self,
        strategies: list[str],
        test_words: list[str],
        metrics: list[ComparisonMetric] | None = None,
        include_statistical_tests: bool = True
    ) -> ComparisonResult:
        """Compare multiple strategies across test words.

        Args:
            strategies: List of strategy names to compare
            test_words: List of words to test strategies against
            metrics: List of metrics to evaluate (default: all)
            include_statistical_tests: Whether to include statistical significance tests

        Returns:
            ComparisonResult with detailed comparison analysis

        Raises:
            WordleAIException: If comparison fails
        """
        if not strategies or not test_words:
            raise WordleAIException("Must provide strategies and test words")

        if metrics is None:
            metrics = list(ComparisonMetric)

        logger.info(f"Comparing {len(strategies)} strategies on {len(test_words)} words")

        # Generate cache key
        cache_key = f"{','.join(sorted(strategies))}_{len(test_words)}_{','.join([m.value for m in metrics])}"

        if cache_key in self._comparison_cache:
            logger.debug("Using cached comparison result")
            return self._comparison_cache[cache_key]

        try:
            # Run strategy tests
            strategy_results = {}
            for strategy_name in strategies:
                logger.info(f"Testing strategy: {strategy_name}")
                result = self._test_strategy(strategy_name, test_words)
                strategy_results[strategy_name] = result

            # Determine winners by metric
            winner_by_metric = self._determine_winners(strategy_results, metrics)

            # Statistical significance testing
            statistical_significance = {}
            if include_statistical_tests and len(strategies) > 1:
                statistical_significance = self._calculate_statistical_significance(strategy_results)

            # Generate recommendations
            recommendations = self._generate_recommendations(strategy_results, winner_by_metric)

            comparison_result = ComparisonResult(
                strategies=strategy_results,
                winner_by_metric=winner_by_metric,
                statistical_significance=statistical_significance,
                recommendations=recommendations
            )

            self._comparison_cache[cache_key] = comparison_result
            logger.info("Strategy comparison completed")

            return comparison_result

        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            raise WordleAIException(f"Strategy comparison failed: {e}") from e

    def _test_strategy(self, strategy_name: str, test_words: list[str]) -> StrategyResult:
        """Test a single strategy against test words.

        Args:
            strategy_name: Name of strategy to test
            test_words: List of words to test against

        Returns:
            StrategyResult with performance metrics
        """
        solver = WordleSolver(strategy=strategy_name)

        attempts_list = []
        times_list = []
        failed_words = []
        attempt_distribution = {}
        total_time = 0.0

        for word in test_words:
            start_time = time.time()

            try:
                result = solver.solve_word(word)
                attempts = result.attempts_used
                success = result.solved

                if success:
                    attempts_list.append(attempts)
                    attempt_distribution[attempts] = attempt_distribution.get(attempts, 0) + 1
                else:
                    failed_words.append(word)

                elapsed_time = time.time() - start_time
                times_list.append(elapsed_time)
                total_time += elapsed_time

            except Exception as e:
                logger.warning(f"Failed to solve {word} with {strategy_name}: {e}")
                failed_words.append(word)
                elapsed_time = time.time() - start_time
                times_list.append(elapsed_time)
                total_time += elapsed_time

        # Calculate metrics
        games_played = len(test_words)
        successful_games = len(attempts_list)
        success_rate = successful_games / games_played if games_played > 0 else 0.0

        average_attempts = np.mean(attempts_list) if attempts_list else 0.0
        median_attempts = np.median(attempts_list) if attempts_list else 0.0
        std_attempts = np.std(attempts_list) if len(attempts_list) > 1 else 0.0
        worst_case_attempts = max(attempts_list) if attempts_list else 0
        average_time = np.mean(times_list) if times_list else 0.0

        return StrategyResult(
            strategy_name=strategy_name,
            success_rate=success_rate,
            average_attempts=average_attempts,
            median_attempts=median_attempts,
            std_attempts=std_attempts,
            total_time=total_time,
            average_time=average_time,
            worst_case_attempts=worst_case_attempts,
            games_played=games_played,
            failed_words=failed_words,
            attempt_distribution=attempt_distribution
        )

    def _determine_winners(
        self,
        strategy_results: dict[str, StrategyResult],
        metrics: list[ComparisonMetric]
    ) -> dict[str, str]:
        """Determine winning strategy for each metric.

        Args:
            strategy_results: Results for each strategy
            metrics: Metrics to evaluate

        Returns:
            Dictionary mapping metric names to winning strategy names
        """
        winner_by_metric = {}

        for metric in metrics:
            best_strategy = None
            best_value = None

            for strategy_name, result in strategy_results.items():
                if metric == ComparisonMetric.SUCCESS_RATE:
                    value = result.success_rate
                    is_better = best_value is None or value > best_value
                elif metric == ComparisonMetric.AVERAGE_ATTEMPTS:
                    value = result.average_attempts
                    is_better = best_value is None or (value < best_value and value > 0)
                elif metric == ComparisonMetric.MEDIAN_ATTEMPTS:
                    value = result.median_attempts
                    is_better = best_value is None or (value < best_value and value > 0)
                elif metric == ComparisonMetric.TIME_EFFICIENCY:
                    value = result.average_time
                    is_better = best_value is None or (value < best_value and value > 0)
                elif metric == ComparisonMetric.WORST_CASE:
                    value = result.worst_case_attempts
                    is_better = best_value is None or (value < best_value and value > 0)
                elif metric == ComparisonMetric.CONSISTENCY:
                    value = result.std_attempts
                    is_better = best_value is None or value < best_value
                else:
                    continue

                if is_better:
                    best_value = value
                    best_strategy = strategy_name

            if best_strategy:
                winner_by_metric[metric.value] = best_strategy

        return winner_by_metric

    def _calculate_statistical_significance(
        self,
        strategy_results: dict[str, StrategyResult]
    ) -> dict[tuple[str, str], dict[str, float]]:
        """Calculate statistical significance between strategy pairs.

        Args:
            strategy_results: Results for each strategy

        Returns:
            Dictionary with statistical test results
        """
        significance_results = {}

        try:
            from scipy import stats

            strategy_names = list(strategy_results.keys())

            for i in range(len(strategy_names)):
                for j in range(i + 1, len(strategy_names)):
                    strategy1 = strategy_names[i]
                    strategy2 = strategy_names[j]

                    result1 = strategy_results[strategy1]
                    result2 = strategy_results[strategy2]

                    # Reconstruct attempt data for statistical tests
                    attempts1 = []
                    for attempts, count in result1.attempt_distribution.items():
                        attempts1.extend([attempts] * count)

                    attempts2 = []
                    for attempts, count in result2.attempt_distribution.items():
                        attempts2.extend([attempts] * count)

                    if len(attempts1) > 0 and len(attempts2) > 0:
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(attempts1, attempts2)

                        # Perform Mann-Whitney U test (non-parametric)
                        u_stat, u_p_value = stats.mannwhitneyu(
                            attempts1, attempts2, alternative='two-sided'
                        )

                        significance_results[(strategy1, strategy2)] = {
                            't_statistic': t_stat,
                            't_p_value': p_value,
                            'u_statistic': u_stat,
                            'u_p_value': u_p_value,
                            'significant_at_05': p_value < 0.05,
                            'significant_at_01': p_value < 0.01
                        }

        except ImportError:
            logger.warning("scipy not available for statistical significance testing")
        except Exception as e:
            logger.warning(f"Statistical significance calculation failed: {e}")

        return significance_results

    def _generate_recommendations(
        self,
        strategy_results: dict[str, StrategyResult],
        winner_by_metric: dict[str, str]
    ) -> list[str]:
        """Generate strategy recommendations based on results.

        Args:
            strategy_results: Results for each strategy
            winner_by_metric: Winners for each metric

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Overall best strategy
        metric_wins = {}
        for winner in winner_by_metric.values():
            metric_wins[winner] = metric_wins.get(winner, 0) + 1

        if metric_wins:
            overall_winner = max(metric_wins.keys(), key=lambda k: metric_wins[k])
            recommendations.append(f"🏆 Overall best strategy: {overall_winner} (wins {metric_wins[overall_winner]} metrics)")

        # Specific use case recommendations
        if ComparisonMetric.SUCCESS_RATE.value in winner_by_metric:
            success_winner = winner_by_metric[ComparisonMetric.SUCCESS_RATE.value]
            success_rate = strategy_results[success_winner].success_rate
            recommendations.append(f"🎯 For maximum success rate: {success_winner} ({success_rate:.1%})")

        if ComparisonMetric.AVERAGE_ATTEMPTS.value in winner_by_metric:
            efficiency_winner = winner_by_metric[ComparisonMetric.AVERAGE_ATTEMPTS.value]
            avg_attempts = strategy_results[efficiency_winner].average_attempts
            recommendations.append(f"⚡ For minimum attempts: {efficiency_winner} ({avg_attempts:.2f} avg)")

        if ComparisonMetric.TIME_EFFICIENCY.value in winner_by_metric:
            speed_winner = winner_by_metric[ComparisonMetric.TIME_EFFICIENCY.value]
            avg_time = strategy_results[speed_winner].average_time
            recommendations.append(f"🚀 For fastest solving: {speed_winner} ({avg_time:.3f}s avg)")

        # Performance gaps analysis
        if len(strategy_results) >= 2:
            best_success = max(r.success_rate for r in strategy_results.values())
            worst_success = min(r.success_rate for r in strategy_results.values())
            success_gap = best_success - worst_success

            if success_gap > 0.05:  # 5% difference
                recommendations.append(f"⚠️  Large success rate gap: {success_gap:.1%} between best and worst strategies")

        return recommendations

    def generate_comparison_report(self, comparison: ComparisonResult) -> str:
        """Generate a comprehensive comparison report.

        Args:
            comparison: Comparison result from compare_strategies()

        Returns:
            Formatted comparison report string
        """
        report_lines = [
            "🔬 STRATEGY COMPARISON ANALYSIS",
            "=" * 50,
            ""
        ]

        # Strategy overview
        report_lines.append("📊 STRATEGY PERFORMANCE OVERVIEW:")
        for strategy_name, result in comparison.strategies.items():
            report_lines.extend([
                f"  {strategy_name.upper()}:",
                f"    Success Rate: {result.success_rate:.1%}",
                f"    Average Attempts: {result.average_attempts:.2f}",
                f"    Median Attempts: {result.median_attempts:.1f}",
                f"    Worst Case: {result.worst_case_attempts} attempts",
                f"    Average Time: {result.average_time:.3f}s",
                f"    Games Played: {result.games_played:,}",
                f"    Failed Words: {len(result.failed_words)}",
                ""
            ])

        # Winners by metric
        if comparison.winner_by_metric:
            report_lines.extend([
                "🏆 WINNERS BY METRIC:",
            ])
            for metric, winner in comparison.winner_by_metric.items():
                report_lines.append(f"  {metric.replace('_', ' ').title()}: {winner}")
            report_lines.append("")

        # Statistical significance
        if comparison.statistical_significance:
            report_lines.extend([
                "📈 STATISTICAL SIGNIFICANCE:",
            ])
            for (strategy1, strategy2), stats in comparison.statistical_significance.items():
                significance = "Yes" if stats['significant_at_05'] else "No"
                report_lines.append(f"  {strategy1} vs {strategy2}: {significance} (p={stats['t_p_value']:.3f})")
            report_lines.append("")

        # Recommendations
        if comparison.recommendations:
            report_lines.extend([
                "💡 RECOMMENDATIONS:",
            ])
            for recommendation in comparison.recommendations:
                report_lines.append(f"  {recommendation}")

        return "\n".join(report_lines)

    def benchmark_strategies(
        self,
        strategies: list[str],
        word_count: int = 1000,
        random_seed: int | None = None
    ) -> ComparisonResult:
        """Run a benchmark comparison of strategies.

        Args:
            strategies: List of strategy names to benchmark
            word_count: Number of words to test (randomly sampled)
            random_seed: Random seed for reproducible results

        Returns:
            ComparisonResult with benchmark results

        Raises:
            WordleAIException: If benchmark fails
        """
        logger.info(f"Running benchmark: {len(strategies)} strategies, {word_count} words")

        try:
            # Get random sample of words
            from ..data.words import WordListManager
            word_manager = WordListManager()
            all_words = word_manager.get_answer_words()

            if random_seed is not None:
                np.random.seed(random_seed)

            if word_count >= len(all_words):
                test_words = all_words
            else:
                test_words = np.random.choice(all_words, size=word_count, replace=False).tolist()

            return self.compare_strategies(
                strategies=strategies,
                test_words=test_words,
                include_statistical_tests=True
            )

        except Exception as e:
            logger.error(f"Strategy benchmark failed: {e}")
            raise WordleAIException(f"Strategy benchmark failed: {e}") from e

    def clear_cache(self) -> None:
        """Clear comparison cache."""
        self._comparison_cache.clear()
        logger.info("Strategy comparison cache cleared")
