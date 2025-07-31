"""Input handling for WORDLE AI terminal interface.

This module provides user input validation and processing
for the terminal-based WORDLE solver interface.
"""

import logging
from collections.abc import Callable

from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.text import Text

logger = logging.getLogger(__name__)


class InputHandler:
    """Handles user input validation and processing.

    Provides methods for getting and validating various types of
    user input for the WORDLE solver interface.
    """

    def __init__(self, console: Console) -> None:
        """Initialize input handler.

        Args:
            console: Rich console instance
        """
        self.console = console
        logger.debug("InputHandler initialized")

    def get_word_input(
        self,
        prompt_text: str = "Enter a 5-letter word",
        allow_empty: bool = False
    ) -> str | None:
        """Get a valid 5-letter word from user.

        Args:
            prompt_text: Prompt text to display
            allow_empty: Whether to allow empty input

        Returns:
            Validated 5-letter word in uppercase, or None if empty and allowed
        """
        while True:
            word = Prompt.ask(prompt_text, default="" if allow_empty else None)

            if not word and allow_empty:
                return None

            if not word:
                self.console.print("[red]❌ Please enter a word.[/red]")
                continue

            word = word.strip().upper()

            if len(word) != 5:
                self.console.print("[red]❌ Word must be exactly 5 letters long.[/red]")
                continue

            if not word.isalpha():
                self.console.print("[red]❌ Word must contain only letters.[/red]")
                continue

            return word

    def get_pattern_input(
        self,
        guess_word: str,
        prompt_text: str | None = None
    ) -> str:
        """Get a valid pattern input from user.

        Args:
            guess_word: The word that was guessed
            prompt_text: Custom prompt text (optional)

        Returns:
            Validated pattern string (G/Y/X format)
        """
        if prompt_text is None:
            prompt_text = f"Enter pattern for '{guess_word}'"

        self.console.print(f"\n[bold]Pattern for '[cyan]{guess_word}[/cyan]':[/bold]")
        self.console.print("🟩 G = Green (correct letter, correct position)")
        self.console.print("🟨 Y = Yellow (correct letter, wrong position)")
        self.console.print("⬜ X = Gray (letter not in word)")
        self.console.print()

        while True:
            pattern = Prompt.ask(
                prompt_text,
                default=""
            ).upper().strip()

            if len(pattern) != 5:
                self.console.print("[red]❌ Pattern must be exactly 5 characters long.[/red]")
                continue

            if not all(c in 'GYX' for c in pattern):
                self.console.print("[red]❌ Pattern must contain only G, Y, or X characters.[/red]")
                continue

            return pattern

    def get_integer_input(
        self,
        prompt_text: str,
        min_value: int | None = None,
        max_value: int | None = None,
        default: int | None = None
    ) -> int:
        """Get a valid integer from user.

        Args:
            prompt_text: Prompt text to display
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            default: Default value if user enters nothing

        Returns:
            Validated integer value
        """
        while True:
            try:
                value = IntPrompt.ask(prompt_text, default=default)

                if min_value is not None and value < min_value:
                    self.console.print(f"[red]❌ Value must be at least {min_value}.[/red]")
                    continue

                if max_value is not None and value > max_value:
                    self.console.print(f"[red]❌ Value must be at most {max_value}.[/red]")
                    continue

                return value

            except (ValueError, TypeError):
                self.console.print("[red]❌ Please enter a valid integer.[/red]")

    def get_choice_input(
        self,
        prompt_text: str,
        choices: list[str],
        default: str | None = None,
        case_sensitive: bool = False
    ) -> str:
        """Get a choice from a list of options.

        Args:
            prompt_text: Prompt text to display
            choices: List of valid choices
            default: Default choice if user enters nothing
            case_sensitive: Whether choices are case sensitive

        Returns:
            Selected choice (in original case from choices list)
        """
        if not case_sensitive:
            choices_lower = [choice.lower() for choice in choices]

        # Display choices
        choices_text = Text()
        for i, choice in enumerate(choices):
            if i > 0:
                choices_text.append(", ")
            choices_text.append(choice, style="cyan")

        self.console.print(f"Choices: {choices_text}")

        while True:
            response = Prompt.ask(prompt_text, default=default)

            if not response:
                if default is not None:
                    return default
                self.console.print("[red]❌ Please make a selection.[/red]")
                continue

            # Find matching choice
            if case_sensitive:
                if response in choices:
                    return response
            else:
                response_lower = response.lower()
                for i, choice_lower in enumerate(choices_lower):
                    if response_lower == choice_lower:
                        return choices[i]

            # Show error with available choices
            choices_display = ", ".join(f"[cyan]{choice}[/cyan]" for choice in choices)
            self.console.print(f"[red]❌ Invalid choice. Please choose from: {choices_display}[/red]")

    def get_yes_no_input(
        self,
        prompt_text: str,
        default: bool | None = None
    ) -> bool:
        """Get a yes/no response from user.

        Args:
            prompt_text: Prompt text to display
            default: Default value (True for yes, False for no, None for no default)

        Returns:
            True for yes, False for no
        """
        default_text = ""
        if default is True:
            default_text = " [Y/n]"
        elif default is False:
            default_text = " [y/N]"
        else:
            default_text = " [y/n]"

        full_prompt = prompt_text + default_text

        while True:
            response = Prompt.ask(full_prompt, default="").strip().lower()

            if not response:
                if default is not None:
                    return default
                self.console.print("[red]❌ Please enter y or n.[/red]")
                continue

            if response in ['y', 'yes', 'true', '1']:
                return True
            elif response in ['n', 'no', 'false', '0']:
                return False
            else:
                self.console.print("[red]❌ Please enter y for yes or n for no.[/red]")

    def get_validated_input(
        self,
        prompt_text: str,
        validator: Callable[[str], bool],
        error_message: str = "Invalid input",
        preprocessor: Callable[[str], str] | None = None,
        allow_empty: bool = False,
        default: str | None = None
    ) -> str | None:
        """Get input with custom validation.

        Args:
            prompt_text: Prompt text to display
            validator: Function that returns True if input is valid
            error_message: Error message to show for invalid input
            preprocessor: Optional function to preprocess input before validation
            allow_empty: Whether to allow empty input
            default: Default value if user enters nothing

        Returns:
            Validated input string, or None if empty and allowed
        """
        while True:
            response = Prompt.ask(prompt_text, default=default)

            if not response:
                if allow_empty:
                    return None
                if default is not None:
                    response = default
                else:
                    self.console.print("[red]❌ Input cannot be empty.[/red]")
                    continue

            # Preprocess if needed
            if preprocessor:
                try:
                    processed_response = preprocessor(response)
                except Exception as e:
                    self.console.print(f"[red]❌ Input processing error: {e}[/red]")
                    continue
            else:
                processed_response = response

            # Validate
            try:
                if validator(processed_response):
                    return processed_response
                else:
                    self.console.print(f"[red]❌ {error_message}[/red]")
            except Exception as e:
                self.console.print(f"[red]❌ Validation error: {e}[/red]")

    def pause_for_user(self, message: str = "Press Enter to continue...") -> None:
        """Pause execution until user presses Enter.

        Args:
            message: Message to display
        """
        Prompt.ask(message, default="", show_default=False)

    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        self.console.clear()


class GameInputHandler(InputHandler):
    """Game-specific input handler for WORDLE AI.

    Extends InputHandler with game-specific input methods
    for interactive WORDLE solving.
    """

    def get_menu_choice(self) -> str:
        """Get main menu choice from user.

        Returns:
            Selected menu option as string
        """
        return Prompt.ask(
            "\n[bold cyan]Select an option[/bold cyan]",
            choices=["0", "1", "2", "3", "4", "5", "6"],
            default="1"
        )

    def get_target_word(self) -> str:
        """Get target word for testing.

        Returns:
            Valid 5-letter target word
        """
        return self.get_word_input("Enter the target word to solve")

    def get_guess_choice(self, recommendation: any) -> str:
        """Get user's choice for the recommended guess.

        Args:
            recommendation: Solver's guess recommendation

        Returns:
            User's choice: "accept", "custom", or "manual"
        """
        self.console.print(f"\n[bold yellow]Recommended guess: [green]{recommendation.guess}[/green][/bold yellow]")

        return Prompt.ask(
            "What would you like to do?",
            choices=["accept", "custom", "manual"],
            default="accept"
        )

    def get_custom_guess(self) -> str:
        """Get custom guess word from user.

        Returns:
            Valid 5-letter guess word
        """
        return self.get_word_input("Enter your custom guess")

    def get_manual_guess(self) -> str:
        """Get manual guess word from user.

        Returns:
            Valid 5-letter guess word
        """
        return self.get_word_input("Enter your manual guess")

    def get_pattern_feedback(self, guess: str) -> str:
        """Get pattern feedback for a guess.

        Args:
            guess: The guessed word

        Returns:
            Pattern string in G/Y/X format
        """
        return self.get_pattern_input(
            guess,
            f"Enter the pattern for '{guess}' (G=🟩, Y=🟨, X=⬜)"
        )

    def get_benchmark_strategies(self) -> list[str]:
        """Get list of strategies to benchmark.

        Returns:
            List of strategy names
        """
        available = ["entropy", "ml", "hybrid", "adaptive"]
        selected = []

        self.console.print("\n[bold]Select strategies to benchmark:[/bold]")
        for strategy in available:
            if self.get_yes_no_input(f"Include {strategy} strategy?", default=True):
                selected.append(strategy)

        return selected or ["hybrid"]  # Default to hybrid if none selected

    def get_advanced_settings(self) -> dict[str, any]:
        """Get advanced solver settings from user.

        Returns:
            Dictionary of advanced settings
        """
        settings = {}

        # ML model settings
        if self.get_yes_no_input("Enable ML predictions?", default=True):
            settings["use_ml"] = True
            settings["ml_weight"] = float(Prompt.ask("ML weight (0.0-1.0)", default="0.3"))

        # Entropy settings
        if self.get_yes_no_input("Enable entropy calculations?", default=True):
            settings["use_entropy"] = True
            settings["entropy_weight"] = float(Prompt.ask("Entropy weight (0.0-1.0)", default="0.7"))

        # Performance settings
        settings["use_caching"] = self.get_yes_no_input("Enable caching?", default=True)
        settings["parallel_processing"] = self.get_yes_no_input("Enable parallel processing?", default=True)

        # Learning settings
        settings["adaptive_learning"] = self.get_yes_no_input("Enable adaptive learning?", default=True)

        return settings
