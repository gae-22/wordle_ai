"""Adaptive learning system for WORDLE solving - Phase 3 implementation.

This module implements sophisticated adaptive learning algorithms that
continuously improve performance based on game outcomes and patterns.
"""

import logging
import pickle
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import MLModelError

logger = logging.getLogger(__name__)


class GameOutcome:
    """Represents the outcome of a single WORDLE game."""

    def __init__(
        self,
        target_word: str,
        guesses: list[str],
        patterns: list[str],
        attempts: int,
        success: bool,
        strategy: str,
        timestamp: datetime | None = None
    ) -> None:
        """Initialize game outcome.

        Args:
            target_word: The target word for the game
            guesses: List of guesses made
            patterns: List of patterns received (G/Y/X format)
            attempts: Number of attempts made
            success: Whether the game was won
            strategy: Strategy used for the game
            timestamp: When the game was played
        """
        self.target_word = target_word.upper()
        self.guesses = [g.upper() for g in guesses]
        self.patterns = patterns
        self.attempts = attempts
        self.success = success
        self.strategy = strategy
        self.timestamp = timestamp or datetime.now()

        # Calculate additional metrics
        self.efficiency = 1.0 / attempts if success else 0.0
        self.difficulty = self._calculate_difficulty()

    def _calculate_difficulty(self) -> float:
        """Calculate the difficulty of the target word.

        Returns:
            Difficulty score (0-1, higher is more difficult)
        """
        # Basic difficulty metrics
        difficulty = 0.0

        # Letter frequency penalty
        common_letters = set('ETAOINSHRDLU')
        uncommon_count = sum(1 for c in self.target_word if c not in common_letters)
        difficulty += uncommon_count * 0.1

        # Repeated letters increase difficulty
        unique_letters = len(set(self.target_word))
        if unique_letters < 5:
            difficulty += (5 - unique_letters) * 0.15

        # Vowel distribution
        vowels = set('AEIOU')
        vowel_count = sum(1 for c in self.target_word if c in vowels)
        if vowel_count < 2 or vowel_count > 3:
            difficulty += 0.1

        return min(1.0, difficulty)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'target_word': self.target_word,
            'guesses': self.guesses,
            'patterns': self.patterns,
            'attempts': self.attempts,
            'success': self.success,
            'strategy': self.strategy,
            'timestamp': self.timestamp.isoformat(),
            'efficiency': self.efficiency,
            'difficulty': self.difficulty
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'GameOutcome':
        """Create GameOutcome from dictionary."""
        timestamp = datetime.fromisoformat(data['timestamp'])
        return cls(
            target_word=data['target_word'],
            guesses=data['guesses'],
            patterns=data['patterns'],
            attempts=data['attempts'],
            success=data['success'],
            strategy=data['strategy'],
            timestamp=timestamp
        )


class AdaptiveLearner(ABC):
    """Abstract base class for adaptive learning algorithms."""

    @abstractmethod
    def update(self, outcome: GameOutcome) -> None:
        """Update the learner with a new game outcome.

        Args:
            outcome: Game outcome to learn from
        """

    @abstractmethod
    def get_adaptation_info(self) -> dict[str, Any]:
        """Get information about current adaptations.

        Returns:
            Dictionary with adaptation statistics
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the adaptive learner."""


class PerformanceTracker:
    """Tracks performance metrics for adaptive learning."""

    def __init__(self, window_size: int = 1000) -> None:
        """Initialize performance tracker.

        Args:
            window_size: Size of sliding window for recent performance
        """
        self.window_size = window_size
        self.recent_outcomes = deque(maxlen=window_size)
        self.strategy_performance: dict[str, list[float]] = defaultdict(list)
        self.word_difficulty: dict[str, float] = {}
        self.pattern_frequencies: dict[str, int] = defaultdict(int)

        # Performance metrics
        self.total_games = 0
        self.total_wins = 0
        self.total_attempts = 0

        logger.debug(f"PerformanceTracker initialized with window size {window_size}")

    def update(self, outcome: GameOutcome) -> None:
        """Update performance tracking with new outcome.

        Args:
            outcome: Game outcome to track
        """
        self.recent_outcomes.append(outcome)
        self.total_games += 1

        if outcome.success:
            self.total_wins += 1
            self.total_attempts += outcome.attempts

        # Track strategy performance
        self.strategy_performance[outcome.strategy].append(outcome.efficiency)

        # Track word difficulty
        self.word_difficulty[outcome.target_word] = outcome.difficulty

        # Track pattern frequencies
        for pattern in outcome.patterns:
            self.pattern_frequencies[pattern] += 1

    def get_recent_performance(self) -> dict[str, float]:
        """Get recent performance metrics.

        Returns:
            Dictionary with recent performance statistics
        """
        if not self.recent_outcomes:
            return {}

        recent_wins = sum(1 for o in self.recent_outcomes if o.success)
        recent_attempts = sum(o.attempts for o in self.recent_outcomes if o.success)
        recent_efficiency = np.mean([o.efficiency for o in self.recent_outcomes])

        return {
            'success_rate': recent_wins / len(self.recent_outcomes),
            'average_attempts': recent_attempts / max(1, recent_wins),
            'average_efficiency': recent_efficiency,
            'games_played': len(self.recent_outcomes)
        }

    def get_strategy_comparison(self) -> dict[str, dict[str, float]]:
        """Compare performance across strategies.

        Returns:
            Dictionary with strategy performance comparisons
        """
        comparison = {}

        for strategy, efficiencies in self.strategy_performance.items():
            if efficiencies:
                comparison[strategy] = {
                    'avg_efficiency': np.mean(efficiencies),
                    'std_efficiency': np.std(efficiencies),
                    'games_played': len(efficiencies),
                    'success_rate': sum(1 for e in efficiencies if e > 0) / len(efficiencies)
                }

        return comparison

    def get_difficulty_analysis(self) -> dict[str, Any]:
        """Analyze word difficulty patterns.

        Returns:
            Dictionary with difficulty analysis
        """
        if not self.word_difficulty:
            return {}

        difficulties = list(self.word_difficulty.values())

        return {
            'avg_difficulty': np.mean(difficulties),
            'difficulty_std': np.std(difficulties),
            'easy_words': sum(1 for d in difficulties if d < 0.3),
            'medium_words': sum(1 for d in difficulties if 0.3 <= d <= 0.7),
            'hard_words': sum(1 for d in difficulties if d > 0.7),
            'total_words': len(difficulties)
        }


class OnlineLearner(AdaptiveLearner):
    """Online learning algorithm that adapts in real-time."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        decay_rate: float = 0.99,
        adaptation_threshold: float = 0.1
    ) -> None:
        """Initialize online learner.

        Args:
            learning_rate: Learning rate for adaptations
            decay_rate: Decay rate for old information
            adaptation_threshold: Threshold for triggering adaptations
        """
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.adaptation_threshold = adaptation_threshold

        self.performance_tracker = PerformanceTracker()
        self.strategy_weights: dict[str, float] = defaultdict(lambda: 1.0)
        self.word_adjustments: dict[str, float] = defaultdict(float)
        self.pattern_rewards: dict[str, float] = defaultdict(float)

        self._lock = threading.Lock()
        logger.debug("OnlineLearner initialized")

    def update(self, outcome: GameOutcome) -> None:
        """Update online learning with new game outcome.

        Args:
            outcome: Game outcome to learn from
        """
        with self._lock:
            self.performance_tracker.update(outcome)

            # Update strategy weights based on performance
            if outcome.success:
                reward = outcome.efficiency
                self.strategy_weights[outcome.strategy] += self.learning_rate * reward
            else:
                penalty = -0.5
                self.strategy_weights[outcome.strategy] += self.learning_rate * penalty

            # Apply decay to prevent old information from dominating
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] *= self.decay_rate

            # Update word-specific adjustments
            adjustment = (outcome.efficiency - 0.5) * self.learning_rate
            self.word_adjustments[outcome.target_word] += adjustment

            # Update pattern rewards
            for pattern in outcome.patterns:
                pattern_reward = outcome.efficiency * self.learning_rate
                self.pattern_rewards[pattern] += pattern_reward

            logger.debug(f"OnlineLearner updated with outcome for {outcome.target_word}")

    def get_strategy_weights(self) -> dict[str, float]:
        """Get current strategy weights.

        Returns:
            Dictionary mapping strategies to their weights
        """
        with self._lock:
            # Normalize weights
            total_weight = sum(self.strategy_weights.values())
            if total_weight > 0:
                return {k: v / total_weight for k, v in self.strategy_weights.items()}
            return dict(self.strategy_weights)

    def get_word_adjustment(self, word: str) -> float:
        """Get adjustment factor for a specific word.

        Args:
            word: Word to get adjustment for

        Returns:
            Adjustment factor (-1 to 1)
        """
        with self._lock:
            return np.clip(self.word_adjustments[word.upper()], -1.0, 1.0)

    def get_pattern_reward(self, pattern: str) -> float:
        """Get reward for a specific pattern.

        Args:
            pattern: Pattern to get reward for

        Returns:
            Pattern reward value
        """
        with self._lock:
            return self.pattern_rewards[pattern]

    def get_adaptation_info(self) -> dict[str, Any]:
        """Get information about current adaptations."""
        with self._lock:
            return {
                'strategy_weights': self.get_strategy_weights(),
                'num_word_adjustments': len(self.word_adjustments),
                'num_pattern_rewards': len(self.pattern_rewards),
                'learning_rate': self.learning_rate,
                'decay_rate': self.decay_rate,
                'recent_performance': self.performance_tracker.get_recent_performance(),
                'strategy_comparison': self.performance_tracker.get_strategy_comparison()
            }

    def reset(self) -> None:
        """Reset the online learner."""
        with self._lock:
            self.strategy_weights.clear()
            self.word_adjustments.clear()
            self.pattern_rewards.clear()
            self.performance_tracker = PerformanceTracker()


class ReinforcementLearner(AdaptiveLearner):
    """Reinforcement learning algorithm for strategy optimization."""

    def __init__(
        self,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_decay: float = 0.995
    ) -> None:
        """Initialize reinforcement learner.

        Args:
            epsilon: Exploration rate
            alpha: Learning rate
            gamma: Discount factor
            epsilon_decay: Epsilon decay rate
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay

        self.q_table: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.state_visits: dict[str, int] = defaultdict(int)
        self.performance_tracker = PerformanceTracker()

        logger.debug("ReinforcementLearner initialized")

    def _get_state(self, outcome: GameOutcome) -> str:
        """Get state representation from game outcome.

        Args:
            outcome: Game outcome

        Returns:
            State string representation
        """
        # Create state based on word characteristics
        word = outcome.target_word
        vowel_count = sum(1 for c in word if c in 'AEIOU')
        unique_letters = len(set(word))
        difficulty_level = 'easy' if outcome.difficulty < 0.3 else 'medium' if outcome.difficulty < 0.7 else 'hard'

        return f"v{vowel_count}_u{unique_letters}_{difficulty_level}"

    def update(self, outcome: GameOutcome) -> None:
        """Update reinforcement learning with new game outcome.

        Args:
            outcome: Game outcome to learn from
        """
        self.performance_tracker.update(outcome)

        state = self._get_state(outcome)
        action = outcome.strategy
        reward = outcome.efficiency if outcome.success else -1.0

        # Update Q-value
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[state].values()) if self.q_table[state] else 0.0

        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

        # Update state visits
        self.state_visits[state] += 1

        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

        logger.debug(f"RL updated: state={state}, action={action}, reward={reward:.3f}, new_q={new_q:.3f}")

    def get_best_strategy(self, word: str) -> str:
        """Get best strategy for a given word.

        Args:
            word: Word to get strategy for

        Returns:
            Best strategy name
        """
        # Create dummy outcome to get state
        dummy_outcome = GameOutcome(word, [], [], 1, True, "dummy")
        state = self._get_state(dummy_outcome)

        if state not in self.q_table or not self.q_table[state]:
            return "entropy"  # Default strategy

        return max(self.q_table[state], key=self.q_table[state].get)

    def get_strategy_probabilities(self, word: str) -> dict[str, float]:
        """Get strategy selection probabilities for a word.

        Args:
            word: Word to get probabilities for

        Returns:
            Dictionary mapping strategies to probabilities
        """
        dummy_outcome = GameOutcome(word, [], [], 1, True, "dummy")
        state = self._get_state(dummy_outcome)

        if state not in self.q_table or not self.q_table[state]:
            return {"entropy": 0.5, "ml": 0.3, "hybrid": 0.2}

        q_values = self.q_table[state]

        # Softmax for probability distribution
        exp_values = {k: np.exp(v / 0.1) for k, v in q_values.items()}  # Temperature = 0.1
        total = sum(exp_values.values())

        return {k: v / total for k, v in exp_values.items()}

    def get_adaptation_info(self) -> dict[str, Any]:
        """Get information about current RL adaptations."""
        return {
            'num_states': len(self.q_table),
            'num_actions': len(set().union(*[actions.keys() for actions in self.q_table.values()])),
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'total_state_visits': sum(self.state_visits.values()),
            'recent_performance': self.performance_tracker.get_recent_performance(),
            'q_table_sample': dict(list(self.q_table.items())[:5])  # Sample of Q-table
        }

    def reset(self) -> None:
        """Reset the reinforcement learner."""
        self.q_table.clear()
        self.state_visits.clear()
        self.performance_tracker = PerformanceTracker()
        self.epsilon = 0.1


class AdaptiveLearningSystem:
    """Main adaptive learning system that coordinates multiple learners."""

    def __init__(
        self,
        save_path: str | None = None,
        auto_save_interval: int = 100
    ) -> None:
        """Initialize adaptive learning system.

        Args:
            save_path: Path to save learning data
            auto_save_interval: Number of games between auto-saves
        """
        self.save_path = Path(save_path) if save_path else None
        self.auto_save_interval = auto_save_interval

        # Initialize learners
        self.online_learner = OnlineLearner()
        self.rl_learner = ReinforcementLearner()

        # Global performance tracker
        self.global_tracker = PerformanceTracker(window_size=5000)

        # Game history
        self.game_history: list[GameOutcome] = []
        self.games_since_save = 0

        logger.info("AdaptiveLearningSystem initialized")

    def update(self, outcome: GameOutcome) -> None:
        """Update all learners with new game outcome.

        Args:
            outcome: Game outcome to learn from
        """
        # Update all learners
        self.online_learner.update(outcome)
        self.rl_learner.update(outcome)
        self.global_tracker.update(outcome)

        # Store in history
        self.game_history.append(outcome)
        self.games_since_save += 1

        # Auto-save if needed
        if self.save_path and self.games_since_save >= self.auto_save_interval:
            self.save_state()
            self.games_since_save = 0

        logger.debug(f"AdaptiveLearningSystem updated with outcome for {outcome.target_word}")

    def get_strategy_recommendation(self, word: str) -> dict[str, str | float]:
        """Get strategy recommendation for a word.

        Args:
            word: Word to get recommendation for

        Returns:
            Dictionary with strategy recommendation and confidence
        """
        # Get recommendations from different learners
        online_weights = self.online_learner.get_strategy_weights()
        # rl_best = self.rl_learner.get_best_strategy(word)  # unused
        rl_probs = self.rl_learner.get_strategy_probabilities(word)

        # Combine recommendations (weighted average)
        combined_scores = defaultdict(float)

        # Weight from online learner
        for strategy, weight in online_weights.items():
            combined_scores[strategy] += weight * 0.4

        # Weight from RL learner
        for strategy, prob in rl_probs.items():
            combined_scores[strategy] += prob * 0.6

        # Get best strategy
        if combined_scores:
            best_strategy = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[best_strategy]
        else:
            best_strategy = "entropy"
            confidence = 0.5

        return {
            'strategy': best_strategy,
            'confidence': confidence,
            'online_weights': online_weights,
            'rl_probabilities': rl_probs
        }

    def get_word_difficulty_prediction(self, word: str) -> float:
        """Predict difficulty for a word based on learned patterns.

        Args:
            word: Word to predict difficulty for

        Returns:
            Predicted difficulty score (0-1)
        """
        # Use historical data if available
        word_upper = word.upper()

        if hasattr(self.global_tracker, 'word_difficulty') and word_upper in self.global_tracker.word_difficulty:
            return self.global_tracker.word_difficulty[word_upper]

        # Otherwise, calculate based on learned patterns
        dummy_outcome = GameOutcome(word, [], [], 1, True, "dummy")
        return dummy_outcome.difficulty

    def get_learning_summary(self) -> dict[str, Any]:
        """Get comprehensive learning summary.

        Returns:
            Dictionary with learning statistics and insights
        """
        return {
            'total_games': len(self.game_history),
            'global_performance': self.global_tracker.get_recent_performance(),
            'strategy_comparison': self.global_tracker.get_strategy_comparison(),
            'difficulty_analysis': self.global_tracker.get_difficulty_analysis(),
            'online_learner': self.online_learner.get_adaptation_info(),
            'rl_learner': self.rl_learner.get_adaptation_info(),
            'recent_games': len([o for o in self.game_history[-100:] if o.timestamp]),
            'learning_trends': self._calculate_learning_trends()
        }

    def _calculate_learning_trends(self) -> dict[str, Any]:
        """Calculate learning trends over time.

        Returns:
            Dictionary with trend analysis
        """
        if len(self.game_history) < 50:
            return {}

        # Calculate trends in recent vs. earlier performance
        recent_games = self.game_history[-100:]
        earlier_games = self.game_history[-500:-100] if len(self.game_history) >= 500 else self.game_history[:-100]

        recent_efficiency = np.mean([g.efficiency for g in recent_games])
        earlier_efficiency = np.mean([g.efficiency for g in earlier_games]) if earlier_games else recent_efficiency

        recent_success = sum(1 for g in recent_games if g.success) / len(recent_games)
        earlier_success = sum(1 for g in earlier_games if g.success) / len(earlier_games) if earlier_games else recent_success

        return {
            'efficiency_trend': recent_efficiency - earlier_efficiency,
            'success_trend': recent_success - earlier_success,
            'improvement_rate': (recent_efficiency - earlier_efficiency) / max(0.01, earlier_efficiency),
            'games_analyzed': {
                'recent': len(recent_games),
                'earlier': len(earlier_games)
            }
        }

    def save_state(self, filepath: str | None = None) -> None:
        """Save learning system state to file.

        Args:
            filepath: Optional custom filepath
        """
        save_path = Path(filepath) if filepath else self.save_path
        if not save_path:
            logger.warning("No save path specified, skipping save")
            return

        try:
            save_data = {
                'game_history': [outcome.to_dict() for outcome in self.game_history],
                'online_learner_data': {
                    'strategy_weights': dict(self.online_learner.strategy_weights),
                    'word_adjustments': dict(self.online_learner.word_adjustments),
                    'pattern_rewards': dict(self.online_learner.pattern_rewards),
                    'learning_rate': self.online_learner.learning_rate,
                    'decay_rate': self.online_learner.decay_rate
                },
                'rl_learner_data': {
                    'q_table': {k: dict(v) for k, v in self.rl_learner.q_table.items()},
                    'state_visits': dict(self.rl_learner.state_visits),
                    'epsilon': self.rl_learner.epsilon,
                    'alpha': self.rl_learner.alpha,
                    'gamma': self.rl_learner.gamma
                },
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }

            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)

            logger.info(f"Learning system state saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save learning system state: {e}")
            raise MLModelError(f"Failed to save learning system state: {e}") from e

    def load_state(self, filepath: str | None = None) -> None:
        """Load learning system state from file.

        Args:
            filepath: Optional custom filepath
        """
        load_path = Path(filepath) if filepath else self.save_path
        if not load_path or not load_path.exists():
            logger.info("No saved state found, starting fresh")
            return

        try:
            with open(load_path, 'rb') as f:
                save_data = pickle.load(f)

            # Restore game history
            self.game_history = [
                GameOutcome.from_dict(outcome_dict)
                for outcome_dict in save_data['game_history']
            ]

            # Restore online learner
            ol_data = save_data['online_learner_data']
            self.online_learner.strategy_weights.update(ol_data['strategy_weights'])
            self.online_learner.word_adjustments.update(ol_data['word_adjustments'])
            self.online_learner.pattern_rewards.update(ol_data['pattern_rewards'])
            self.online_learner.learning_rate = ol_data['learning_rate']
            self.online_learner.decay_rate = ol_data['decay_rate']

            # Restore RL learner
            rl_data = save_data['rl_learner_data']
            for state, actions in rl_data['q_table'].items():
                self.rl_learner.q_table[state].update(actions)
            self.rl_learner.state_visits.update(rl_data['state_visits'])
            self.rl_learner.epsilon = rl_data['epsilon']
            self.rl_learner.alpha = rl_data['alpha']
            self.rl_learner.gamma = rl_data['gamma']

            # Rebuild performance trackers
            for outcome in self.game_history:
                self.global_tracker.update(outcome)
                self.online_learner.performance_tracker.update(outcome)
                self.rl_learner.performance_tracker.update(outcome)

            logger.info(f"Loaded learning system state with {len(self.game_history)} games")

        except Exception as e:
            logger.error(f"Failed to load learning system state: {e}")
            raise MLModelError(f"Failed to load learning system state: {e}") from e

    def reset_all(self) -> None:
        """Reset all learning components."""
        self.online_learner.reset()
        self.rl_learner.reset()
        self.global_tracker = PerformanceTracker(window_size=5000)
        self.game_history.clear()
        self.games_since_save = 0

        logger.info("All learning components reset")
