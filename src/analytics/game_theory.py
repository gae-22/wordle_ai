"""Game theory optimization for WORDLE AI.

This module applies game theory principles to optimize WORDLE solving strategies,
including minimax algorithms, Nash equilibrium analysis, and strategic decision making.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .. import WordleAIException
from ..data.patterns import PatternMatcher
from ..solver.entropy import EntropyCalculator

logger = logging.getLogger(__name__)


class GameStrategy(Enum):
    """Game theory strategies for WORDLE."""
    MINIMAX = "minimax"
    MAXIMIN = "maximin"
    NASH_EQUILIBRIUM = "nash_equilibrium"
    REGRET_MINIMIZATION = "regret_minimization"
    EXPECTED_VALUE = "expected_value"


@dataclass
class GameState:
    """Represents a game state in WORDLE."""
    possible_words: list[str]
    previous_guesses: list[str]
    guess_patterns: list[str]
    turn_number: int
    max_turns: int = 6


@dataclass
class StrategyResult:
    """Result of a game theory strategy analysis."""
    recommended_guess: str
    expected_outcome: float
    worst_case_outcome: float
    best_case_outcome: float
    strategy_confidence: float
    decision_tree: dict[str, Any] | None = None


@dataclass
class OptimizationResult:
    """Results of game theory optimization."""
    optimal_strategy: GameStrategy
    performance_metrics: dict[str, float]
    strategy_comparison: dict[GameStrategy, float]
    nash_equilibrium: dict[str, float] | None
    recommendations: list[str]


class GameTheoryOptimizer:
    """Game theory-based optimization for WORDLE solving.

    Applies various game theory principles to find optimal strategies
    and decision-making approaches for WORDLE puzzles.
    """

    def __init__(self, entropy_calculator: EntropyCalculator) -> None:
        """Initialize game theory optimizer.

        Args:
            entropy_calculator: Entropy calculator for information analysis
        """
        logger.debug("Initializing GameTheoryOptimizer")

        self._entropy_calculator = entropy_calculator
        self._pattern_matcher = PatternMatcher()

        # Strategy weights for multi-criteria optimization
        self._strategy_weights = {
            'expected_value': 0.4,
            'worst_case': 0.3,
            'information_gain': 0.2,
            'risk_tolerance': 0.1
        }

        logger.debug("GameTheoryOptimizer initialized")

    def find_optimal_guess(
        self,
        game_state: GameState,
        strategy: GameStrategy = GameStrategy.MINIMAX,
        candidate_guesses: list[str] | None = None
    ) -> StrategyResult:
        """Find optimal guess using game theory strategy.

        Args:
            game_state: Current game state
            strategy: Game theory strategy to use
            candidate_guesses: List of candidate guesses (default: all possible words)

        Returns:
            StrategyResult with optimal guess and analysis

        Raises:
            WordleAIException: If optimization fails
        """
        if not game_state.possible_words:
            raise WordleAIException("No possible words remaining")

        if game_state.turn_number >= game_state.max_turns:
            raise WordleAIException("Maximum turns exceeded")

        logger.debug(f"Finding optimal guess using {strategy.value} strategy")

        if candidate_guesses is None:
            candidate_guesses = game_state.possible_words

        try:
            if strategy == GameStrategy.MINIMAX:
                return self._minimax_strategy(game_state, candidate_guesses)
            elif strategy == GameStrategy.MAXIMIN:
                return self._maximin_strategy(game_state, candidate_guesses)
            elif strategy == GameStrategy.NASH_EQUILIBRIUM:
                return self._nash_equilibrium_strategy(game_state, candidate_guesses)
            elif strategy == GameStrategy.REGRET_MINIMIZATION:
                return self._regret_minimization_strategy(game_state, candidate_guesses)
            elif strategy == GameStrategy.EXPECTED_VALUE:
                return self._expected_value_strategy(game_state, candidate_guesses)
            else:
                raise WordleAIException(f"Unknown strategy: {strategy}")

        except Exception as e:
            logger.error(f"Optimal guess finding failed: {e}")
            raise WordleAIException(f"Optimal guess finding failed: {e}") from e

    def _minimax_strategy(
        self,
        game_state: GameState,
        candidate_guesses: list[str]
    ) -> StrategyResult:
        """Apply minimax strategy to find optimal guess.

        Args:
            game_state: Current game state
            candidate_guesses: Candidate guesses to evaluate

        Returns:
            StrategyResult with minimax optimal guess
        """
        best_guess = None
        best_score = float('-inf')
        decision_tree = {}

        for guess in candidate_guesses:
            # Calculate worst-case scenario for this guess
            worst_case_score = self._calculate_worst_case_outcome(
                guess, game_state.possible_words
            )

            # Calculate expected outcome
            expected_score = self._calculate_expected_outcome(
                guess, game_state.possible_words
            )

            # Minimax score: maximize the minimum guaranteed outcome
            minimax_score = worst_case_score * 0.7 + expected_score * 0.3

            decision_tree[guess] = {
                'worst_case': worst_case_score,
                'expected': expected_score,
                'minimax_score': minimax_score
            }

            if minimax_score > best_score:
                best_score = minimax_score
                best_guess = guess

        if not best_guess:
            best_guess = candidate_guesses[0]

        return StrategyResult(
            recommended_guess=best_guess,
            expected_outcome=decision_tree[best_guess]['expected'],
            worst_case_outcome=decision_tree[best_guess]['worst_case'],
            best_case_outcome=1.0,  # Best case is always solving in one guess
            strategy_confidence=self._calculate_strategy_confidence(decision_tree, best_guess),
            decision_tree=decision_tree
        )

    def _maximin_strategy(
        self,
        game_state: GameState,
        candidate_guesses: list[str]
    ) -> StrategyResult:
        """Apply maximin strategy (maximize minimum outcome).

        Args:
            game_state: Current game state
            candidate_guesses: Candidate guesses to evaluate

        Returns:
            StrategyResult with maximin optimal guess
        """
        best_guess = None
        best_worst_case = float('-inf')
        decision_tree = {}

        for guess in candidate_guesses:
            worst_case_score = self._calculate_worst_case_outcome(
                guess, game_state.possible_words
            )

            expected_score = self._calculate_expected_outcome(
                guess, game_state.possible_words
            )

            decision_tree[guess] = {
                'worst_case': worst_case_score,
                'expected': expected_score
            }

            if worst_case_score > best_worst_case:
                best_worst_case = worst_case_score
                best_guess = guess

        if not best_guess:
            best_guess = candidate_guesses[0]

        return StrategyResult(
            recommended_guess=best_guess,
            expected_outcome=decision_tree[best_guess]['expected'],
            worst_case_outcome=best_worst_case,
            best_case_outcome=1.0,
            strategy_confidence=self._calculate_strategy_confidence(decision_tree, best_guess),
            decision_tree=decision_tree
        )

    def _nash_equilibrium_strategy(
        self,
        game_state: GameState,
        candidate_guesses: list[str]
    ) -> StrategyResult:
        """Apply Nash equilibrium strategy.

        Args:
            game_state: Current game state
            candidate_guesses: Candidate guesses to evaluate

        Returns:
            StrategyResult with Nash equilibrium strategy
        """
        # Simplified Nash equilibrium approximation
        # In practice, this would involve solving a more complex game matrix

        best_guess = None
        best_equilibrium_score = float('-inf')
        decision_tree = {}

        for guess in candidate_guesses:
            # Calculate information gain (player's utility)
            info_gain = self._entropy_calculator.calculate_guess_entropy(
                guess, game_state.possible_words
            )

            # Calculate opponent's (adversary's) counter-utility
            # Adversary wants to minimize our information gain
            opponent_utility = self._calculate_opponent_utility(
                guess, game_state.possible_words
            )

            # Nash equilibrium approximation: balance our utility vs opponent's
            equilibrium_score = info_gain - 0.5 * opponent_utility

            decision_tree[guess] = {
                'info_gain': info_gain,
                'opponent_utility': opponent_utility,
                'equilibrium_score': equilibrium_score
            }

            if equilibrium_score > best_equilibrium_score:
                best_equilibrium_score = equilibrium_score
                best_guess = guess

        if not best_guess:
            best_guess = candidate_guesses[0]

        expected_outcome = self._calculate_expected_outcome(
            best_guess, game_state.possible_words
        )
        worst_case_outcome = self._calculate_worst_case_outcome(
            best_guess, game_state.possible_words
        )

        return StrategyResult(
            recommended_guess=best_guess,
            expected_outcome=expected_outcome,
            worst_case_outcome=worst_case_outcome,
            best_case_outcome=1.0,
            strategy_confidence=self._calculate_strategy_confidence(decision_tree, best_guess),
            decision_tree=decision_tree
        )

    def _regret_minimization_strategy(
        self,
        game_state: GameState,
        candidate_guesses: list[str]
    ) -> StrategyResult:
        """Apply regret minimization strategy.

        Args:
            game_state: Current game state
            candidate_guesses: Candidate guesses to evaluate

        Returns:
            StrategyResult with regret-minimizing guess
        """
        best_guess = None
        min_regret = float('inf')
        decision_tree = {}

        # Calculate the best possible outcome for each scenario
        scenario_best_outcomes = {}
        for target_word in game_state.possible_words:
            best_outcome_for_target = 0
            for guess in candidate_guesses:
                outcome = self._calculate_outcome_for_target(guess, target_word)
                best_outcome_for_target = max(best_outcome_for_target, outcome)
            scenario_best_outcomes[target_word] = best_outcome_for_target

        for guess in candidate_guesses:
            max_regret = 0
            total_regret = 0

            for target_word in game_state.possible_words:
                actual_outcome = self._calculate_outcome_for_target(guess, target_word)
                best_possible = scenario_best_outcomes[target_word]
                regret = best_possible - actual_outcome

                max_regret = max(max_regret, regret)
                total_regret += regret

            avg_regret = total_regret / len(game_state.possible_words)

            decision_tree[guess] = {
                'max_regret': max_regret,
                'avg_regret': avg_regret,
                'total_regret': total_regret
            }

            if max_regret < min_regret:
                min_regret = max_regret
                best_guess = guess

        if not best_guess:
            best_guess = candidate_guesses[0]

        expected_outcome = self._calculate_expected_outcome(
            best_guess, game_state.possible_words
        )
        worst_case_outcome = self._calculate_worst_case_outcome(
            best_guess, game_state.possible_words
        )

        return StrategyResult(
            recommended_guess=best_guess,
            expected_outcome=expected_outcome,
            worst_case_outcome=worst_case_outcome,
            best_case_outcome=1.0,
            strategy_confidence=self._calculate_strategy_confidence(decision_tree, best_guess),
            decision_tree=decision_tree
        )

    def _expected_value_strategy(
        self,
        game_state: GameState,
        candidate_guesses: list[str]
    ) -> StrategyResult:
        """Apply expected value maximization strategy.

        Args:
            game_state: Current game state
            candidate_guesses: Candidate guesses to evaluate

        Returns:
            StrategyResult with expected value optimal guess
        """
        best_guess = None
        best_expected_value = float('-inf')
        decision_tree = {}

        for guess in candidate_guesses:
            expected_value = self._calculate_expected_outcome(
                guess, game_state.possible_words
            )

            # Add information gain component
            info_gain = self._entropy_calculator.calculate_guess_entropy(
                guess, game_state.possible_words
            )

            # Normalize information gain to [0, 1]
            max_possible_entropy = math.log2(len(game_state.possible_words))
            normalized_info_gain = info_gain / max_possible_entropy if max_possible_entropy > 0 else 0

            # Combined expected value
            combined_expected_value = (
                0.7 * expected_value + 0.3 * normalized_info_gain
            )

            decision_tree[guess] = {
                'expected_outcome': expected_value,
                'info_gain': info_gain,
                'combined_value': combined_expected_value
            }

            if combined_expected_value > best_expected_value:
                best_expected_value = combined_expected_value
                best_guess = guess

        if not best_guess:
            best_guess = candidate_guesses[0]

        worst_case_outcome = self._calculate_worst_case_outcome(
            best_guess, game_state.possible_words
        )

        return StrategyResult(
            recommended_guess=best_guess,
            expected_outcome=decision_tree[best_guess]['expected_outcome'],
            worst_case_outcome=worst_case_outcome,
            best_case_outcome=1.0,
            strategy_confidence=self._calculate_strategy_confidence(decision_tree, best_guess),
            decision_tree=decision_tree
        )

    def _calculate_worst_case_outcome(
        self,
        guess: str,
        possible_words: list[str]
    ) -> float:
        """Calculate worst-case outcome for a guess.

        Args:
            guess: Guess word
            possible_words: List of possible target words

        Returns:
            Worst-case outcome score (0-1, higher is better)
        """
        pattern_groups = defaultdict(list)

        # Group words by the pattern they would produce with this guess
        for target_word in possible_words:
            pattern = self._pattern_matcher.get_pattern(guess, target_word)
            pattern_groups[pattern].append(target_word)

        # Worst case is the largest remaining group after this guess
        max_remaining = max(len(group) for group in pattern_groups.values()) if pattern_groups else 0

        # Convert to score (fewer remaining words is better)
        if len(possible_words) <= 1:
            return 1.0

        reduction_ratio = 1.0 - (max_remaining / len(possible_words))
        return max(0.0, reduction_ratio)

    def _calculate_expected_outcome(
        self,
        guess: str,
        possible_words: list[str]
    ) -> float:
        """Calculate expected outcome for a guess.

        Args:
            guess: Guess word
            possible_words: List of possible target words

        Returns:
            Expected outcome score (0-1, higher is better)
        """
        if not possible_words:
            return 0.0

        pattern_groups = defaultdict(list)

        # Group words by pattern
        for target_word in possible_words:
            pattern = self._pattern_matcher.get_pattern(guess, target_word)
            pattern_groups[pattern].append(target_word)

        # Calculate expected remaining words
        total_words = len(possible_words)
        expected_remaining = 0.0

        for _, words in pattern_groups.items():
            probability = len(words) / total_words
            remaining_after_pattern = len(words)
            expected_remaining += probability * remaining_after_pattern

        # Convert to score
        if total_words <= 1:
            return 1.0

        reduction_ratio = 1.0 - (expected_remaining / total_words)
        return max(0.0, reduction_ratio)

    def _calculate_opponent_utility(
        self,
        guess: str,
        possible_words: list[str]
    ) -> float:
        """Calculate opponent's utility (adversarial perspective).

        Args:
            guess: Guess word
            possible_words: List of possible target words

        Returns:
            Opponent's utility score
        """
        # Opponent wants to maximize our remaining uncertainty
        # This is the inverse of information gain
        info_gain = self._entropy_calculator.calculate_guess_entropy(guess, possible_words)
        max_possible_entropy = math.log2(len(possible_words)) if len(possible_words) > 1 else 1.0

        # Opponent's utility is higher when our information gain is lower
        return max_possible_entropy - info_gain

    def _calculate_outcome_for_target(self, guess: str, target_word: str) -> float:
        """Calculate outcome for a specific target word.

        Args:
            guess: Guess word
            target_word: Target word

        Returns:
            Outcome score for this specific target
        """
        if guess == target_word:
            return 1.0  # Perfect outcome

        # Calculate how much information this guess provides about the target
        pattern = self._pattern_matcher.get_pattern(guess, target_word)

        # Score based on pattern quality
        green_count = pattern.count('G')
        yellow_count = pattern.count('Y')
        gray_count = pattern.count('X')

        # Higher score for more informative patterns
        info_score = (green_count * 0.4 + yellow_count * 0.3 + gray_count * 0.1) / 5
        return info_score

    def _calculate_strategy_confidence(
        self,
        decision_tree: dict[str, Any],
        best_guess: str
    ) -> float:
        """Calculate confidence in strategy decision.

        Args:
            decision_tree: Decision analysis for all candidates
            best_guess: Selected best guess

        Returns:
            Confidence score (0-1)
        """
        if not decision_tree or best_guess not in decision_tree:
            return 0.5

        # Get the scores for all guesses
        scores = []
        best_score = None

        for guess, analysis in decision_tree.items():
            # Use the main score from the analysis
            if 'minimax_score' in analysis:
                score = analysis['minimax_score']
            elif 'equilibrium_score' in analysis:
                score = analysis['equilibrium_score']
            elif 'combined_value' in analysis:
                score = analysis['combined_value']
            else:
                score = analysis.get('expected', 0)

            scores.append(score)
            if guess == best_guess:
                best_score = score

        if not scores or best_score is None:
            return 0.5

        # Confidence based on how much better the best choice is
        scores.sort(reverse=True)
        if len(scores) < 2:
            return 0.9

        second_best = scores[1]
        if second_best == 0:
            return 0.9

        gap = (best_score - second_best) / abs(second_best) if second_best != 0 else 1.0
        confidence = min(0.9, 0.5 + gap * 0.4)

        return max(0.1, confidence)

    def optimize_strategy_mix(
        self,
        game_scenarios: list[GameState],
        strategies: list[GameStrategy]
    ) -> OptimizationResult:
        """Optimize mix of strategies across different game scenarios.

        Args:
            game_scenarios: List of game scenarios to test
            strategies: List of strategies to evaluate

        Returns:
            OptimizationResult with optimal strategy recommendations

        Raises:
            WordleAIException: If optimization fails
        """
        if not game_scenarios or not strategies:
            raise WordleAIException("Must provide game scenarios and strategies")

        logger.info(f"Optimizing strategy mix across {len(game_scenarios)} scenarios")

        try:
            strategy_performance = {}
            detailed_results = {}

            # Evaluate each strategy across all scenarios
            for strategy in strategies:
                total_score = 0.0
                scenario_scores = []

                for scenario in game_scenarios:
                    try:
                        result = self.find_optimal_guess(scenario, strategy)
                        score = (
                            result.expected_outcome * 0.5 +
                            result.worst_case_outcome * 0.3 +
                            result.strategy_confidence * 0.2
                        )
                        total_score += score
                        scenario_scores.append(score)
                    except Exception as e:
                        logger.warning(f"Failed to evaluate {strategy.value} on scenario: {e}")
                        scenario_scores.append(0.0)

                avg_score = total_score / len(game_scenarios) if game_scenarios else 0.0
                strategy_performance[strategy] = avg_score
                detailed_results[strategy] = {
                    'average_score': avg_score,
                    'scenario_scores': scenario_scores,
                    'std_score': np.std(scenario_scores) if scenario_scores else 0.0
                }

            # Find optimal strategy
            optimal_strategy = max(strategy_performance.keys(), key=lambda k: strategy_performance[k])

            # Calculate Nash equilibrium approximation
            nash_equilibrium = self._approximate_nash_equilibrium(strategy_performance)

            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(
                strategy_performance, detailed_results
            )

            performance_metrics = {
                'best_strategy_score': strategy_performance[optimal_strategy],
                'strategy_diversity': np.std(list(strategy_performance.values())),
                'scenarios_tested': len(game_scenarios),
                'strategies_tested': len(strategies)
            }

            return OptimizationResult(
                optimal_strategy=optimal_strategy,
                performance_metrics=performance_metrics,
                strategy_comparison=strategy_performance,
                nash_equilibrium=nash_equilibrium,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            raise WordleAIException(f"Strategy optimization failed: {e}") from e

    def _approximate_nash_equilibrium(
        self,
        strategy_performance: dict[GameStrategy, float]
    ) -> dict[str, float]:
        """Approximate Nash equilibrium for strategy mixing.

        Args:
            strategy_performance: Performance scores for each strategy

        Returns:
            Dictionary with equilibrium probabilities
        """
        # Simple Nash equilibrium approximation using normalized scores
        total_score = sum(strategy_performance.values())

        if total_score == 0:
            # Uniform distribution if all strategies perform equally
            prob = 1.0 / len(strategy_performance)
            return {strategy.value: prob for strategy in strategy_performance}

        # Probability proportional to performance
        equilibrium = {}
        for strategy, score in strategy_performance.items():
            equilibrium[strategy.value] = max(0.01, score / total_score)

        # Normalize to ensure probabilities sum to 1
        total_prob = sum(equilibrium.values())
        if total_prob > 0:
            equilibrium = {k: v / total_prob for k, v in equilibrium.items()}

        return equilibrium

    def _generate_optimization_recommendations(
        self,
        strategy_performance: dict[GameStrategy, float],
        detailed_results: dict[GameStrategy, dict[str, Any]]
    ) -> list[str]:
        """Generate optimization recommendations.

        Args:
            strategy_performance: Strategy performance scores
            detailed_results: Detailed results for each strategy

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Best overall strategy
        best_strategy = max(strategy_performance.keys(), key=lambda k: strategy_performance[k])
        best_score = strategy_performance[best_strategy]
        recommendations.append(f"🏆 Best overall strategy: {best_strategy.value} (score: {best_score:.3f})")

        # Consistency analysis
        most_consistent = min(
            detailed_results.keys(),
            key=lambda k: detailed_results[k]['std_score']
        )
        recommendations.append(f"📊 Most consistent strategy: {most_consistent.value}")

        # Performance gaps
        worst_strategy = min(strategy_performance.keys(), key=lambda k: strategy_performance[k])
        performance_gap = best_score - strategy_performance[worst_strategy]
        if performance_gap > 0.1:
            recommendations.append(f"⚠️  Large performance gap: {performance_gap:.3f} between best and worst strategies")

        # Strategy mixing recommendation
        top_strategies = sorted(
            strategy_performance.keys(),
            key=lambda k: strategy_performance[k],
            reverse=True
        )[:2]

        if len(top_strategies) == 2:
            score_diff = strategy_performance[top_strategies[0]] - strategy_performance[top_strategies[1]]
            if score_diff < 0.05:  # Close performance
                recommendations.append(
                    f"🔄 Consider mixing {top_strategies[0].value} and {top_strategies[1].value} strategies"
                )

        return recommendations
