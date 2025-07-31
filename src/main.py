#!/usr/bin/env python3
"""Main entry point for WORDLE AI Solver.

This module provides the interactive terminal-based interface for the WORDLE AI solver,
offering an immersive command-line experience with Rich-based TUI components.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Confirm, Prompt

from .solver.engine import WordleSolver
from .ui.display import WordleDisplay
from .ui.input import GameInputHandler
from .utils.helpers import setup_logging

logger = logging.getLogger(__name__)


class WordleApp:
    """Main WORDLE AI application with interactive TUI.

    This class orchestrates the interactive terminal experience, providing
    menu-driven navigation and rich visual feedback.
    """

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the WORDLE AI application.

        Args:
            console: Optional Rich console instance
        """
        self.console = console or Console()
        self.display = WordleDisplay(self.console)
        self.input_handler = GameInputHandler(self.console)
        self.solver: WordleSolver | None = None

        logger.info("WordleApp initialized")

    def run_interactive_mode(self) -> None:
        """Run the main interactive TUI mode."""
        self.display.show_welcome()

        while True:
            self.display.show_main_menu()
            choice = self.input_handler.get_menu_choice()

            try:
                if choice == "1":
                    self._run_single_game()
                elif choice == "2":
                    self._run_benchmark()
                elif choice == "3":
                    self._configure_solver()
                elif choice == "4":
                    self._show_analytics()
                elif choice == "5":
                    self._train_models()
                elif choice == "6":
                    self._show_settings()
                elif choice == "0":
                    self._exit_application()
                    break
                else:
                    self.console.print("[red]Invalid choice. Please try again.[/red]")
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Operation cancelled.[/yellow]")
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                self.console.print(f"[red]Error: {e}[/red]")

    def _run_single_game(self) -> None:
        """Run a single interactive game."""
        if not self.solver:
            self._initialize_default_solver()

        # Reset game state
        self.solver.reset_game()

        self.console.print("\n[bold cyan]🎯 Starting Interactive WORDLE Game[/bold cyan]")

        # Ask if user wants to provide target word or play blind
        mode = Prompt.ask(
            "Game mode",
            choices=["interactive", "target", "random"],
            default="interactive"
        )

        target_word = None
        if mode == "target":
            target_word = self.input_handler.get_target_word()
        elif mode == "random":
            # Let solver pick a random word
            target_word = self.solver.get_random_target()
            self.console.print(f"[dim]Random target selected (hidden)[/dim]")

        # Run the game
        game_result = self._play_interactive_game(target_word)

        # Show results
        self.display.show_game_result(game_result)

        # Ask if user wants to save results
        if Confirm.ask("Save game results for learning?", default=True):
            self._save_game_result(game_result)

    def _play_interactive_game(self, target_word: str | None) -> Any:
        """Play an interactive game with user feedback.

        Args:
            target_word: Optional target word for validation

        Returns:
            GameResult object with game statistics
        """
        if not self.solver:
            raise RuntimeError("Solver not initialized")

        guesses = []
        attempt = 0
        max_attempts = 6

        self.display.show_game_board()

        while attempt < max_attempts:
            attempt += 1

            # Get solver's recommendation
            with self.console.status(f"[bold blue]Calculating optimal guess {attempt}/6..."):
                recommendation = self.solver.get_best_guess()

            self.display.show_guess_recommendation(recommendation.guess, recommendation.remaining_words)

            # Get user's choice
            user_choice = self.input_handler.get_guess_choice(recommendation)

            if user_choice == "accept":
                guess = recommendation.guess
            elif user_choice == "custom":
                guess = self.input_handler.get_custom_guess()
            else:  # manual
                guess = self.input_handler.get_manual_guess()

            # Get pattern feedback
            if target_word:
                pattern = self.solver.calculate_pattern(guess, target_word)
                self.console.print(f"[dim]Pattern calculated: {pattern}[/dim]")
            else:
                pattern = self.input_handler.get_pattern_feedback(guess)

            # Process the guess
            result = self.solver.process_guess(guess, pattern)
            guesses.append(result)

            # Update display
            self.display.show_guess_result(result, attempt)

            # Check if solved
            if pattern == "GGGGG":
                self.console.print(f"[bold green]🎉 Congratulations! Solved in {attempt} attempts![/bold green]")
                return self._create_game_result(target_word, guesses, True, attempt)

            # Show remaining possibilities
            if result.remaining_words <= 10:
                remaining = self.solver.get_remaining_words()[:10]
                self.display.show_remaining_words(remaining)

        # Game not solved
        self.console.print("[red]❌ Game not solved within 6 attempts.[/red]")
        return self._create_game_result(target_word, guesses, False, max_attempts)

    def _run_benchmark(self) -> None:
        """Run benchmark testing on word list."""
        if not self.solver:
            self._initialize_default_solver()

        self.console.print("\n[bold yellow]📊 Running Benchmark Analysis[/bold yellow]")

        # Configuration options
        word_count = int(Prompt.ask("Number of words to test", default="100"))
        strategies = self.input_handler.get_benchmark_strategies()

        with self.console.status("[bold blue]Running benchmark..."):
            results = self.solver.run_benchmark(
                word_count=word_count,
                strategies=strategies
            )

        self.display.show_benchmark_results(results)

    def _configure_solver(self) -> None:
        """Configure solver settings."""
        self.console.print("\n[bold blue]⚙️ Solver Configuration[/bold blue]")

        strategy = Prompt.ask(
            "Select strategy",
            choices=["entropy", "ml", "hybrid", "adaptive"],
            default="hybrid"
        )

        max_attempts = int(Prompt.ask("Max attempts per game", default="6"))

        # Advanced settings
        if Confirm.ask("Configure advanced settings?", default=False):
            settings = self.input_handler.get_advanced_settings()
        else:
            settings = {}

        # Initialize new solver
        self.solver = WordleSolver(
            strategy=strategy,
            max_attempts=max_attempts,
            **settings
        )

        self.console.print(f"[green]✓ Solver configured with {strategy} strategy[/green]")

    def _show_analytics(self) -> None:
        """Show analytics and statistics."""
        if not self.solver:
            self._initialize_default_solver()

        self.console.print("\n[bold magenta]📈 Analytics Dashboard[/bold magenta]")

        analytics_type = Prompt.ask(
            "Select analytics type",
            choices=["strategy", "difficulty", "performance", "learning"],
            default="strategy"
        )

        with self.console.status("[bold blue]Generating analytics..."):
            if analytics_type == "strategy":
                results = self.solver.analyze_strategies()
            elif analytics_type == "difficulty":
                results = self.solver.analyze_word_difficulty()
            elif analytics_type == "performance":
                results = self.solver.get_performance_report()
            else:  # learning
                results = self.solver.get_learning_summary()

        self.display.show_analytics_results(results, analytics_type)

    def _train_models(self) -> None:
        """Train or retrain ML models."""
        if not self.solver:
            self._initialize_default_solver()

        self.console.print("\n[bold green]🤖 Machine Learning Training[/bold green]")

        training_type = Prompt.ask(
            "Training type",
            choices=["quick", "full", "neural", "adaptive"],
            default="quick"
        )

        with self.console.status("[bold blue]Training models..."):
            training_results = self.solver.train_models(training_type)

        self.display.show_training_results(training_results)

    def _show_settings(self) -> None:
        """Show current settings and allow modifications."""
        self.console.print("\n[bold cyan]⚙️ Current Settings[/bold cyan]")

        if self.solver:
            settings = self.solver.get_current_settings()
            self.display.show_settings(settings)
        else:
            self.console.print("[yellow]No solver initialized yet.[/yellow]")

        if Confirm.ask("Modify settings?", default=False):
            self._configure_solver()

    def _initialize_default_solver(self) -> None:
        """Initialize solver with default settings."""
        self.console.print("[dim]Initializing solver with default settings...[/dim]")
        self.solver = WordleSolver(strategy="hybrid")
        logger.info("Default solver initialized")

    def _create_game_result(self, target_word: str | None, guesses: list, solved: bool, attempts: int) -> Any:
        """Create GameResult object from game data."""
        from .solver.engine import GameResult

        total_time = sum(guess.processing_time for guess in guesses)

        return GameResult(
            target_word=target_word,
            guesses=guesses,
            solved=solved,
            attempts=attempts,
            total_time=total_time,
            strategy_used=self.solver.current_strategy if self.solver else "unknown"
        )

    def _save_game_result(self, game_result: Any) -> None:
        """Save game result for future learning."""
        if self.solver:
            self.solver.save_game_result(game_result)
            self.console.print("[green]✓ Game result saved for learning[/green]")

    def _exit_application(self) -> None:
        """Clean exit with goodbye message."""
        self.display.show_goodbye()
        logger.info("Application terminated by user")

    def _solve_word_interactive(self, target_word: str) -> None:
        """Solve a specific word in interactive TUI mode.

        Args:
            target_word: The word to solve
        """
        if not self.solver:
            self._initialize_default_solver()

        self.console.print(f"\n[bold cyan]🎯 Solving word: {target_word}[/bold cyan]")

        # Run the game with known target
        game_result = self._play_interactive_game(target_word)

        # Show results
        self.display.show_game_result(game_result)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="WORDLE AI Solver - Intelligent Terminal-based Puzzle Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wordle-ai                    # Interactive TUI mode (default)
  wordle-ai --benchmark        # Run benchmark tests
  wordle-ai --solve AROSE      # Solve specific word
  wordle-ai --train            # Train ML models
  wordle-ai --strategy ml      # Use ML strategy
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        default=True,
        help="Interactive TUI mode (default)"
    )
    mode_group.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run benchmark testing"
    )
    mode_group.add_argument(
        "--solve", "-s",
        type=str,
        metavar="WORD",
        help="Solve specific word"
    )
    mode_group.add_argument(
        "--train", "-t",
        action="store_true",
        help="Train ML models"
    )

    # Configuration options
    parser.add_argument(
        "--strategy",
        choices=["entropy", "ml", "hybrid", "adaptive"],
        default="hybrid",
        help="Solving strategy (default: hybrid)"
    )
    parser.add_argument(
        "--word-list",
        type=Path,
        help="Path to custom word list file"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=6,
        help="Maximum attempts per game (default: 6)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
    )

    return parser


def main() -> int:
    """Main entry point for the WORDLE AI application.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level=args.log_level,
        quiet=args.quiet
    )

    # Create console
    console = Console(quiet=args.quiet)

    try:
        app = WordleApp(console)

        # Default to interactive mode if no specific mode selected
        if not any([args.benchmark, args.solve, args.train]):
            app.run_interactive_mode()
        elif args.benchmark:
            # Run benchmark mode (TUI-based)
            app._run_benchmark()
        elif args.solve:
            # Solve specific word (TUI-based)
            app._solve_word_interactive(args.solve)
        elif args.train:
            # Train models (TUI-based)
            app._train_models()

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Application interrupted by user.[/yellow]")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
