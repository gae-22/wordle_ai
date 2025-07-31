"""Rich-based display components for WORDLE AI.

This module provides terminal UI components using the Rich library
for beautiful, interactive WORDLE solving displays.
"""

import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)


class WordleDisplay:
    """Rich-based terminal display for WORDLE AI solver.

    Provides beautiful, interactive terminal interface for the WORDLE solver
    with colorized output, progress indicators, and user interaction.
    """

    def __init__(self, console: Console) -> None:
        """Initialize display with Rich console.

        Args:
            console: Rich console instance
        """
        self.console = console
        self._colors = {
            'green': 'green',
            'yellow': 'yellow',
            'gray': 'dim white',
            'correct': 'bold green',
            'partial': 'bold yellow',
            'wrong': 'dim white on black'
        }

        logger.debug("WordleDisplay initialized")

    def show_welcome(self) -> None:
        """Display welcome message and instructions."""
        welcome_text = Text()
        welcome_text.append("🎯 WORDLE AI SOLVER 🎯\n", style="bold cyan")
        welcome_text.append("Intelligent Terminal-based Puzzle Solver\n\n", style="cyan")
        welcome_text.append("Features:\n", style="bold white")
        welcome_text.append("• Information Theory (Entropy Calculations)\n", style="white")
        welcome_text.append("• Machine Learning Predictions\n", style="white")
        welcome_text.append("• Adaptive Learning Strategies\n", style="white")
        welcome_text.append("• Real-time Performance Analytics\n\n", style="white")

        panel = Panel(
            welcome_text,
            title="Welcome",
            border_style="cyan",
            padding=(1, 2)
        )

        self.console.print(panel)
        self.console.print()

    def show_goodbye(self) -> None:
        """Display goodbye message."""
        goodbye_text = Text()
        goodbye_text.append("Thanks for using WORDLE AI Solver! 🎯\n", style="bold cyan")
        goodbye_text.append("Happy puzzling! 🧩", style="cyan")

        panel = Panel(
            goodbye_text,
            title="Goodbye",
            border_style="cyan",
            padding=(1, 2)
        )

        self.console.print(panel)

    def show_guess_recommendation(self, guess: str, remaining_words: int) -> None:
        """Display guess recommendation with statistics.

        Args:
            guess: Recommended guess word
            remaining_words: Number of remaining possible words
        """
        # Create guess display
        guess_display = Text()
        for letter in guess:
            guess_display.append(f" {letter} ", style="bold white on blue")

        # Create info table
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_row("💡 Recommended Guess:", guess_display)
        info_table.add_row("🎯 Remaining Words:", f"[bold yellow]{remaining_words}[/bold yellow]")

        panel = Panel(
            info_table,
            title="🤖 AI Recommendation",
            border_style="blue",
            padding=(1, 1)
        )

        self.console.print(panel)

    def get_pattern_feedback(self, guess: str) -> str:
        """Get pattern feedback from user for a guess.

        Args:
            guess: The guessed word

        Returns:
            Pattern string (G/Y/X format)
        """
        self.console.print(f"\n[bold]Enter feedback for guess '[cyan]{guess}[/cyan]':[/bold]")
        self.console.print("🟩 = G (Green - correct letter, correct position)")
        self.console.print("🟨 = Y (Yellow - correct letter, wrong position)")
        self.console.print("⬜ = X (Gray - letter not in word)")
        self.console.print()

        while True:
            pattern = Prompt.ask(
                "Feedback pattern",
                default="",
                show_default=False
            ).upper().strip()

            if len(pattern) == 5 and all(c in 'GYX' for c in pattern):
                return pattern

            self.console.print("[red]❌ Invalid pattern! Please enter exactly 5 characters using G, Y, or X.[/red]")

    def show_game_result(self, result: Any) -> None:
        """Display game result summary.

        Args:
            result: GameResult object with solving statistics
        """
        # Create result display
        if result.solved:
            title = "🎉 Puzzle Solved!"
            title_style = "bold green"
            status_text = f"[bold green]✅ Solved in {result.attempts} attempts![/bold green]"
        else:
            title = "❌ Puzzle Not Solved"
            title_style = "bold red"
            status_text = f"[bold red]❌ Unable to solve in {result.attempts} attempts[/bold red]"

        # Create stats table
        stats_table = Table(show_header=False, box=None, padding=(0, 1))
        stats_table.add_row("Status:", status_text)
        stats_table.add_row("Attempts:", f"[yellow]{result.attempts}[/yellow]")
        stats_table.add_row("Total Time:", f"[cyan]{result.total_time:.2f}s[/cyan]")
        stats_table.add_row("Strategy:", f"[magenta]{result.strategy_used}[/magenta]")

        if result.target_word:
            stats_table.add_row("Target Word:", f"[bold cyan]{result.target_word}[/bold cyan]")

        # Show guess history
        if result.guesses:
            self.console.print("\n[bold]Guess History:[/bold]")
            guess_table = Table(show_header=True, header_style="bold blue")
            guess_table.add_column("#", style="dim", width=3)
            guess_table.add_column("Guess", style="bold")
            guess_table.add_column("Pattern", width=15)
            guess_table.add_column("Remaining", justify="right")
            guess_table.add_column("Entropy", justify="right")
            guess_table.add_column("Time", justify="right")

            for i, guess_result in enumerate(result.guesses, 1):
                pattern_display = self._format_pattern(guess_result.pattern)
                guess_table.add_row(
                    str(i),
                    guess_result.guess,
                    pattern_display,
                    str(guess_result.remaining_words),
                    f"{guess_result.entropy:.2f}",
                    f"{guess_result.processing_time:.3f}s"
                )

            self.console.print(guess_table)

        # Main result panel
        panel = Panel(
            stats_table,
            title=title,
            border_style=title_style.split()[1] if ' ' in title_style else title_style,
            padding=(1, 2)
        )

        self.console.print(panel)

    def show_solve_result(self, result: Any) -> None:
        """Display result for solving a specific word.

        Args:
            result: GameResult object
        """
        self.show_game_result(result)

    def show_benchmark_results(self, results: Any) -> None:
        """Display benchmark test results.

        Args:
            results: BenchmarkResult object
        """
        # Summary stats
        summary_table = Table(show_header=False, box=None, padding=(0, 1))
        summary_table.add_row("Total Words Tested:", f"[bold yellow]{results.total_words}[/bold yellow]")
        summary_table.add_row("Words Solved:", f"[bold green]{results.solved_count}[/bold green]")
        summary_table.add_row("Success Rate:", f"[bold cyan]{results.success_rate:.1f}%[/bold cyan]")
        summary_table.add_row("Average Attempts:", f"[bold magenta]{results.average_attempts:.2f}[/bold magenta]")
        summary_table.add_row("Total Time:", f"[bold white]{results.total_time:.1f}s[/bold white]")

        panel = Panel(
            summary_table,
            title="📊 Benchmark Results",
            border_style="green",
            padding=(1, 2)
        )

        self.console.print(panel)

        # Strategy performance breakdown
        if results.strategy_performance:
            self.console.print("\n[bold]Strategy Performance:[/bold]")
            strategy_table = Table(show_header=True, header_style="bold blue")
            strategy_table.add_column("Strategy", style="bold")
            strategy_table.add_column("Success Rate", justify="right")
            strategy_table.add_column("Avg Attempts", justify="right")
            strategy_table.add_column("Avg Time", justify="right")

            for strategy, metrics in results.strategy_performance.items():
                strategy_table.add_row(
                    strategy,
                    f"{metrics['success_rate']:.1f}%",
                    f"{metrics['avg_attempts']:.2f}",
                    f"{metrics['avg_time']:.3f}s"
                )

            self.console.print(strategy_table)

    def show_error(self, message: str) -> None:
        """Display error message.

        Args:
            message: Error message to display
        """
        panel = Panel(
            f"[bold red]{message}[/bold red]",
            title="❌ Error",
            border_style="red",
            padding=(1, 2)
        )

        self.console.print(panel)

    def show_progress(self, description: str, total: int | None = None) -> Progress:
        """Create and return a progress indicator.

        Args:
            description: Description of the progress
            total: Total number of steps (None for spinner)

        Returns:
            Progress object for updates
        """
        if total is None:
            # Spinner for indeterminate progress
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            )
        else:
            # Progress bar for determinate progress
            progress = Progress(console=self.console)

        return progress

    def confirm(self, question: str) -> bool:
        """Ask user a yes/no question.

        Args:
            question: Question to ask

        Returns:
            True if user confirms, False otherwise
        """
        return Confirm.ask(question, default=True)

    def _format_pattern(self, pattern: str) -> Text:
        """Format pattern string with colors.

        Args:
            pattern: Pattern string (G/Y/X)

        Returns:
            Colored Text object
        """
        text = Text()

        for char in pattern:
            if char == 'G':
                text.append("🟩", style="green")
            elif char == 'Y':
                text.append("🟨", style="yellow")
            else:  # X or other
                text.append("⬜", style="dim white")

        return text

    # Phase 3 ML Training Display Methods

    def show_training_start(self) -> None:
        """Display training start message."""
        self.console.print(Panel(
            Text("🚀 Starting Phase 3 ML Model Training\n\n" +
                 "This will train multiple ML models including:\n" +
                 "• Linear regression models\n" +
                 "• Deep neural networks\n" +
                 "• Random forest models\n" +
                 "• Gradient boosting models\n\n" +
                 "Training may take several minutes...",
                 style="cyan"),
            title="[bold blue]ML Training Pipeline[/bold blue]",
            border_style="blue"
        ))

    def show_training_results(self, results: dict[str, Any]) -> None:
        """Display comprehensive training results.

        Args:
            results: Training results dictionary
        """
        # Create summary table
        table = Table(title="[bold green]Training Results Summary[/bold green]")
        table.add_column("Model Type", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Test MSE", style="yellow")
        table.add_column("Test R²", style="blue")
        table.add_column("Training Time", style="magenta")

        model_training = results.get('model_training', {})

        for model_type, model_result in model_training.items():
            if 'error' in model_result:
                status = "[red]Failed[/red]"
                mse = "N/A"
                r2 = "N/A"
                time_str = "N/A"
            else:
                status = "[green]Success[/green]"
                test_metrics = model_result.get('test_metrics', {})
                mse = f"{test_metrics.get('mse', 0):.6f}"
                r2 = f"{test_metrics.get('r2', 0):.4f}"
                time_str = f"{model_result.get('training_time', 0):.2f}s"

            table.add_row(model_type, status, mse, r2, time_str)

        self.console.print(table)

        # Show best model
        best_model = results.get('pipeline_summary', {}).get('best_model', {})
        if best_model and best_model.get('model_type') != 'none':
            self.console.print(f"\n[bold green]🏆 Best Model:[/bold green] {best_model['model_type']}")
            self.console.print(f"  Test MSE: {best_model.get('test_mse', 0):.6f}")
            self.console.print(f"  Test R²: {best_model.get('test_r2', 0):.4f}")

        # Show report path
        report_path = results.get('report_path')
        if report_path:
            self.console.print(f"\n[dim]📊 Detailed report saved to: {report_path}[/dim]")

    def show_neural_training_start(self) -> None:
        """Display neural network training start message."""
        self.console.print(Panel(
            Text("🧠 Neural Network Training\n\n" +
                 "Training sophisticated neural networks:\n" +
                 "• Deep feedforward network with residual connections\n" +
                 "• Convolutional neural network for pattern recognition\n" +
                 "• Advanced optimization with early stopping\n\n" +
                 "This process uses PyTorch and may take some time...",
                 style="purple"),
            title="[bold purple]Neural Network Training[/bold purple]",
            border_style="purple"
        ))

    def show_neural_training_results(self, results: dict[str, Any]) -> None:
        """Display neural network training results.

        Args:
            results: Neural training results
        """
        table = Table(title="[bold purple]Neural Network Results[/bold purple]")
        table.add_column("Architecture", style="cyan")
        table.add_column("Parameters", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Device", style="blue")

        for _model_name, model_info in results.items():
            architecture = model_info.get('architecture', 'Unknown')
            num_params = model_info.get('num_parameters', 0)
            is_trained = model_info.get('is_trained', False)
            device = model_info.get('device', 'cpu')

            status = "[green]Trained[/green]" if is_trained else "[red]Failed[/red]"
            params_str = f"{num_params:,}" if num_params > 0 else "N/A"

            table.add_row(architecture, params_str, status, device)

        self.console.print(table)

    def show_adaptive_learning_start(self) -> None:
        """Display adaptive learning demonstration start."""
        self.console.print(Panel(
            Text("🎯 Adaptive Learning System\n\n" +
                 "Demonstrating intelligent adaptation:\n" +
                 "• Online learning with real-time updates\n" +
                 "• Reinforcement learning for strategy optimization\n" +
                 "• Performance tracking and trend analysis\n" +
                 "• Strategy weight adaptation\n\n" +
                 "Simulating games to show learning in action...",
                 style="green"),
            title="[bold green]Adaptive Learning Demo[/bold green]",
            border_style="green"
        ))

    def show_adaptive_learning_results(self, summary: dict[str, Any]) -> None:
        """Display adaptive learning results.

        Args:
            summary: Learning system summary
        """
        # Performance summary
        global_perf = summary.get('global_performance', {})
        if global_perf:
            perf_table = Table(title="[bold green]Performance Summary[/bold green]")
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="yellow")

            perf_table.add_row("Success Rate", f"{global_perf.get('success_rate', 0):.1%}")
            perf_table.add_row("Average Attempts", f"{global_perf.get('average_attempts', 0):.2f}")
            perf_table.add_row("Average Efficiency", f"{global_perf.get('average_efficiency', 0):.3f}")
            perf_table.add_row("Games Played", str(global_perf.get('games_played', 0)))

            self.console.print(perf_table)

        # Strategy comparison
        strategy_comp = summary.get('strategy_comparison', {})
        if strategy_comp:
            strat_table = Table(title="[bold blue]Strategy Performance[/bold blue]")
            strat_table.add_column("Strategy", style="cyan")
            strat_table.add_column("Success Rate", style="green")
            strat_table.add_column("Avg Efficiency", style="yellow")
            strat_table.add_column("Games", style="blue")

            for strategy, stats in strategy_comp.items():
                success_rate = f"{stats.get('success_rate', 0):.1%}"
                avg_eff = f"{stats.get('avg_efficiency', 0):.3f}"
                games = str(stats.get('games_played', 0))

                strat_table.add_row(strategy, success_rate, avg_eff, games)

            self.console.print(strat_table)

        # Learning trends
        trends = summary.get('learning_trends', {})
        if trends:
            self.console.print("\n[bold magenta]Learning Trends:[/bold magenta]")
            efficiency_trend = trends.get('efficiency_trend', 0)
            success_trend = trends.get('success_trend', 0)

            eff_symbol = "📈" if efficiency_trend > 0 else "📉" if efficiency_trend < 0 else "➡️"
            succ_symbol = "📈" if success_trend > 0 else "📉" if success_trend < 0 else "➡️"

            self.console.print(f"  {eff_symbol} Efficiency: {efficiency_trend:+.3f}")
            self.console.print(f"  {succ_symbol} Success Rate: {success_trend:+.1%}")

        # Online learner info
        online_info = summary.get('online_learner', {})
        if online_info:
            weights = online_info.get('strategy_weights', {})
            if weights:
                self.console.print("\n[bold yellow]Current Strategy Weights:[/bold yellow]")
                for strategy, weight in weights.items():
                    bar_length = int(weight * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    self.console.print(f"  {strategy:>8}: {bar} {weight:.3f}")

        # Show total games processed
        total_games = summary.get('total_games', 0)
        self.console.print(f"\n[dim]📊 Processed {total_games} total games in learning system[/dim]")

    def show_performance_optimization_results(self, report: dict[str, Any]) -> None:
        """Display performance optimization results.

        Args:
            report: Performance optimization report
        """
        self.console.print(Panel(
            "[bold cyan]Performance Optimization Report[/bold cyan]",
            border_style="cyan"
        ))

        # Optimization status
        opt_status = report.get('optimization_status', {})
        status_table = Table(title="Optimization Features")
        status_table.add_column("Feature", style="cyan")
        status_table.add_column("Status", style="green")

        for feature, enabled in opt_status.items():
            status = "[green]✓ Enabled[/green]" if enabled else "[red]✗ Disabled[/red]"
            status_table.add_row(feature.replace('_', ' ').title(), status)

        self.console.print(status_table)

        # Cache statistics
        cache_stats = report.get('cache_stats', {})
        if cache_stats:
            self.console.print("\n[bold yellow]Cache Performance:[/bold yellow]")
            self.console.print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
            self.console.print(f"  Cache Size: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}")

        # Memory statistics
        memory_stats = report.get('memory_stats', {})
        if memory_stats:
            self.console.print("\n[bold red]Memory Usage:[/bold red]")
            self.console.print(f"  Current: {memory_stats.get('current_memory_mb', 0):.1f} MB")
            self.console.print(f"  Available: {memory_stats.get('available_memory_mb', 0):.1f} MB")
            self.console.print(f"  Usage: {memory_stats.get('memory_percent', 0):.1f}%")
