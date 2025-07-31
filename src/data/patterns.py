"""Pattern matching and feedback processing for WORDLE.

This module handles the generation and analysis of WORDLE feedback patterns,
including converting game feedback into filterable patterns and filtering
word lists based on constraints.
"""

import logging
from collections import defaultdict

from .. import PatternError

logger = logging.getLogger(__name__)


class PatternMatcher:
    """Handles WORDLE pattern generation and word filtering.

    Processes feedback patterns (Green, Yellow, Gray) and filters word lists
    based on the constraints they impose.
    """

    def __init__(self) -> None:
        """Initialize pattern matcher."""
        logger.debug("PatternMatcher initialized")

    def generate_pattern(self, guess: str, target: str) -> str:
        """Generate WORDLE feedback pattern for a guess against target.

        Args:
            guess: The guessed word
            target: The target/answer word

        Returns:
            Pattern string using G(green), Y(yellow), X(gray)

        Raises:
            PatternError: If words are invalid
        """
        if len(guess) != 5 or len(target) != 5:
            raise PatternError("Both guess and target must be 5 letters")

        if not (guess.isalpha() and target.isalpha()):
            raise PatternError("Words must contain only letters")

        guess = guess.upper()
        target = target.upper()

        pattern = ['X'] * 5  # Start with all gray
        target_chars = list(target)

        # First pass: Mark exact matches (green)
        for i in range(5):
            if guess[i] == target[i]:
                pattern[i] = 'G'
                target_chars[i] = None  # Mark as used

        # Second pass: Mark wrong position matches (yellow)
        for i in range(5):
            if pattern[i] == 'X':  # Only check non-green positions
                guess_char = guess[i]
                for j in range(5):
                    if target_chars[j] == guess_char:
                        pattern[i] = 'Y'
                        target_chars[j] = None  # Mark as used
                        break

        result = ''.join(pattern)
        logger.debug(f"Pattern for '{guess}' vs '{target}': {result}")
        return result

    def filter_words(self, words: list[str], guess: str, pattern: str) -> list[str]:
        """Filter word list based on guess and pattern feedback.

        Args:
            words: List of words to filter
            guess: The guessed word
            pattern: Feedback pattern (G/Y/X)

        Returns:
            Filtered list of words that match the constraints

        Raises:
            PatternError: If pattern is invalid
        """
        if len(pattern) != 5:
            raise PatternError("Pattern must be 5 characters")

        if not all(c in 'GYX' for c in pattern):
            raise PatternError("Pattern must contain only G, Y, X characters")

        if len(guess) != 5:
            raise PatternError("Guess must be 5 letters")

        guess = guess.upper()
        pattern = pattern.upper()

        logger.debug(f"Filtering {len(words)} words with guess '{guess}' and pattern '{pattern}'")

        filtered_words = []

        for word in words:
            if self._word_matches_pattern(word.upper(), guess, pattern):
                filtered_words.append(word)

        logger.debug(f"Filtered to {len(filtered_words)} words")
        return filtered_words

    def _word_matches_pattern(self, word: str, guess: str, pattern: str) -> bool:
        """Check if a word matches the given guess and pattern.

        Args:
            word: Word to check
            guess: The guessed word
            pattern: Expected pattern

        Returns:
            True if word would produce the same pattern
        """
        try:
            actual_pattern = self.generate_pattern(guess, word)
            return actual_pattern == pattern
        except PatternError:
            return False

    def get_pattern_constraints(self, guess: str, pattern: str) -> dict[str, any]:
        """Extract constraints from a guess-pattern combination.

        Args:
            guess: The guessed word
            pattern: Feedback pattern

        Returns:
            Dictionary containing extracted constraints

        Raises:
            PatternError: If inputs are invalid
        """
        if len(guess) != 5 or len(pattern) != 5:
            raise PatternError("Guess and pattern must be 5 characters")

        guess = guess.upper()
        pattern = pattern.upper()

        constraints = {
            'required_positions': {},  # position -> letter (green)
            'required_letters': set(),  # letters that must be in word (yellow/green)
            'forbidden_letters': set(),  # letters that cannot be in word (gray)
            'forbidden_positions': defaultdict(set),  # position -> set of forbidden letters
        }

        # Count letters in guess for yellow constraint processing
        guess_letter_counts = defaultdict(int)
        for letter in guess:
            guess_letter_counts[letter] += 1

        # Process each position
        for i, (letter, color) in enumerate(zip(guess, pattern, strict=False)):
            if color == 'G':
                # Green: letter must be in this exact position
                constraints['required_positions'][i] = letter
                constraints['required_letters'].add(letter)

            elif color == 'Y':
                # Yellow: letter must be in word but not in this position
                constraints['required_letters'].add(letter)
                constraints['forbidden_positions'][i].add(letter)

            elif color == 'X':
                # Gray: letter should not be in word (with exceptions for duplicates)
                # Only mark as forbidden if all instances of this letter are gray
                letter_positions = [j for j, c in enumerate(guess) if c == letter]
                letter_patterns = [pattern[j] for j in letter_positions]

                if all(p == 'X' for p in letter_patterns):
                    constraints['forbidden_letters'].add(letter)
                else:
                    # Letter appears elsewhere as Y/G, so just forbidden in this position
                    constraints['forbidden_positions'][i].add(letter)

        return constraints

    def validate_word_against_constraints(
        self,
        word: str,
        constraints: dict[str, any]
    ) -> bool:
        """Check if a word satisfies pattern constraints.

        Args:
            word: Word to validate
            constraints: Constraints from get_pattern_constraints

        Returns:
            True if word satisfies all constraints
        """
        word = word.upper()

        # Check required positions (green)
        for pos, required_letter in constraints['required_positions'].items():
            if pos >= len(word) or word[pos] != required_letter:
                return False

        # Check required letters (yellow/green)
        for required_letter in constraints['required_letters']:
            if required_letter not in word:
                return False

        # Check forbidden letters (gray)
        for forbidden_letter in constraints['forbidden_letters']:
            if forbidden_letter in word:
                return False

        # Check forbidden positions (yellow)
        for pos, forbidden_letters in constraints['forbidden_positions'].items():
            if pos < len(word) and word[pos] in forbidden_letters:
                return False

        return True

    def analyze_pattern_distribution(
        self,
        guess: str,
        possible_words: list[str]
    ) -> dict[str, int]:
        """Analyze how many words would produce each pattern.

        Args:
            guess: The guessed word
            possible_words: List of possible target words

        Returns:
            Dictionary mapping patterns to word counts
        """
        pattern_counts: dict[str, int] = defaultdict(int)

        for word in possible_words:
            try:
                pattern = self.generate_pattern(guess, word)
                pattern_counts[pattern] += 1
            except PatternError:
                continue  # Skip invalid words

        return dict(pattern_counts)

    def get_most_informative_positions(
        self,
        possible_words: list[str]
    ) -> list[tuple[int, float]]:
        """Find positions with highest letter diversity.

        Args:
            possible_words: List of words to analyze

        Returns:
            List of (position, diversity_score) tuples sorted by diversity
        """
        position_diversity = []

        for pos in range(5):
            letters_at_pos = set()
            for word in possible_words:
                if pos < len(word):
                    letters_at_pos.add(word[pos].upper())

            # Diversity score is number of unique letters
            diversity = len(letters_at_pos)
            position_diversity.append((pos, diversity))

        # Sort by diversity (descending)
        position_diversity.sort(key=lambda x: x[1], reverse=True)
        return position_diversity
