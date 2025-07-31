"""Word list management for WORDLE solver.

This module handles loading, validating, and managing word lists used
for WORDLE solving, including valid guesses and possible answers.
"""

import logging
import urllib.error
import urllib.request
from pathlib import Path

from .. import WordListError

logger = logging.getLogger(__name__)


class WordListManager:
    """Manages WORDLE word lists and validation.

    Handles both valid guess words and possible answer words,
    with support for custom word lists and validation.
    """

    def __init__(self, custom_word_list: Path | None = None) -> None:
        """Initialize word list manager.

        Args:
            custom_word_list: Optional path to custom word list file

        Raises:
            WordListError: If word list loading fails
        """
        logger.info("Initializing WordListManager")

        self._valid_words: set[str] = set()
        self._answer_words: set[str] = set()

        try:
            if custom_word_list:
                self._load_custom_word_list(custom_word_list)
            else:
                self._load_default_word_lists()

            logger.info(f"Loaded {len(self._valid_words)} valid words, {len(self._answer_words)} answer words")

        except Exception as e:
            logger.error(f"Failed to initialize word lists: {e}")
            raise WordListError(f"Word list initialization failed: {e}") from e

    def get_valid_words(self) -> list[str]:
        """Get list of all valid guess words.

        Returns:
            Sorted list of valid 5-letter words
        """
        return sorted(list(self._valid_words))

    def get_answer_words(self) -> list[str]:
        """Get list of possible answer words.

        Returns:
            Sorted list of possible answer words
        """
        return sorted(list(self._answer_words))

    def is_valid_word(self, word: str) -> bool:
        """Check if a word is valid for guessing.

        Args:
            word: Word to validate (will be converted to uppercase)

        Returns:
            True if word is valid for guessing
        """
        return word.upper() in self._valid_words

    def is_possible_answer(self, word: str) -> bool:
        """Check if a word is a possible answer.

        Args:
            word: Word to check (will be converted to uppercase)

        Returns:
            True if word could be an answer
        """
        return word.upper() in self._answer_words

    def add_word(self, word: str, is_answer: bool = True) -> None:
        """Add a word to the word lists.

        Args:
            word: Word to add (will be converted to uppercase)
            is_answer: Whether word can be an answer

        Raises:
            WordListError: If word is invalid
        """
        word = word.upper()

        if len(word) != 5:
            raise WordListError(f"Word must be 5 letters: {word}")

        if not word.isalpha():
            raise WordListError(f"Word must contain only letters: {word}")

        self._valid_words.add(word)
        if is_answer:
            self._answer_words.add(word)

        logger.debug(f"Added word: {word} (answer: {is_answer})")

    def remove_word(self, word: str) -> None:
        """Remove a word from word lists.

        Args:
            word: Word to remove (will be converted to uppercase)
        """
        word = word.upper()
        self._valid_words.discard(word)
        self._answer_words.discard(word)
        logger.debug(f"Removed word: {word}")

    def _load_custom_word_list(self, word_list_path: Path) -> None:
        """Load words from custom file.

        Args:
            word_list_path: Path to word list file

        Raises:
            WordListError: If file cannot be loaded
        """
        logger.info(f"Loading custom word list: {word_list_path}")

        try:
            with open(word_list_path, encoding='utf-8') as f:
                words = [line.strip().upper() for line in f if line.strip()]

            if not words:
                raise WordListError(f"Empty word list: {word_list_path}")

            # Validate words
            valid_words = []
            for word in words:
                if len(word) == 5 and word.isalpha():
                    valid_words.append(word)
                else:
                    logger.warning(f"Skipping invalid word: {word}")

            if not valid_words:
                raise WordListError("No valid 5-letter words found")

            self._valid_words = set(valid_words)
            self._answer_words = set(valid_words)  # Assume all can be answers

        except FileNotFoundError:
            raise WordListError(f"Word list file not found: {word_list_path}") from None
        except Exception as e:
            raise WordListError(f"Failed to load word list: {e}") from e

    def _load_default_word_lists(self) -> None:
        """Load default WORDLE word lists.

        First tries to download from official sources, falls back to embedded lists.
        """
        logger.info("Loading default word lists")

        try:
            # Try to download from web sources
            self._download_word_lists()
        except Exception as e:
            logger.warning(f"Failed to download word lists from web: {e}")
            logger.info("Falling back to embedded word lists")
            self._load_embedded_word_lists()

    def _download_word_lists(self) -> None:
        """Download word lists from official WORDLE sources.

        Raises:
            WordListError: If download fails
        """
        logger.info("Downloading word lists from official sources")

        # 1. WORDLE公式解答リスト(確実性の高いソース)
        answer_sources = [
            "https://raw.githubusercontent.com/3b1b/videos/master/_2022/wordle/data/possible_words.txt",
            "https://raw.githubusercontent.com/charlesreid1/five-letter-words/master/sgb-words.txt",
        ]

        # 2. WORDLE有効推測単語リスト(確実性の高いソース)
        guess_sources = [
            "https://raw.githubusercontent.com/3b1b/videos/master/_2022/wordle/data/allowed_words.txt",
            "https://raw.githubusercontent.com/charlesreid1/five-letter-words/master/sgb-words.txt",
        ]

        # Download answer words
        answer_words = set()
        for source in answer_sources:
            try:
                words = self._download_from_url(source)
                answer_words.update(words)
                logger.info(f"Downloaded {len(words)} answer words from {source}")
            except Exception as e:
                logger.warning(f"Failed to download from {source}: {e}")

        # Download valid guess words
        valid_words = set()
        for source in guess_sources:
            try:
                words = self._download_from_url(source)
                valid_words.update(words)
                logger.info(f"Downloaded {len(words)} valid words from {source}")
            except Exception as e:
                logger.warning(f"Failed to download from {source}: {e}")

        if not answer_words and not valid_words:
            raise WordListError("Failed to download any word lists")

        # Ensure answer words are subset of valid words
        if answer_words and valid_words:
            valid_words.update(answer_words)

        # Set the word lists
        self._valid_words = valid_words if valid_words else answer_words
        self._answer_words = answer_words if answer_words else valid_words

        logger.info(f"Successfully downloaded {len(self._valid_words)} valid words, {len(self._answer_words)} answer words")

    def _download_from_url(self, url: str, timeout: int = 10) -> set[str]:
        """Download and parse word list from URL.

        Args:
            url: URL to download from
            timeout: Request timeout in seconds

        Returns:
            Set of valid 5-letter words

        Raises:
            WordListError: If download or parsing fails
        """
        logger.debug(f"Downloading from {url}")

        try:
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'WORDLE-AI-Solver/1.0'}
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read().decode('utf-8')

            words = set()
            for line in content.strip().split('\n'):
                word = line.strip().upper()
                if len(word) == 5 and word.isalpha():
                    words.add(word)
                elif word and len(word) != 5:
                    logger.debug(f"Skipping non-5-letter word: {word}")

            logger.debug(f"Parsed {len(words)} valid words from {url}")
            return words

        except urllib.error.URLError as e:
            raise WordListError(f"Network error downloading from {url}: {e}") from e
        except Exception as e:
            raise WordListError(f"Error processing {url}: {e}") from e

    def _load_embedded_word_lists(self) -> None:
        """Load embedded WORDLE word lists as fallback.

        Uses embedded word lists when web download fails.
        """
        logger.info("Loading default word lists")

        # Default valid guess words (common 5-letter words)
        valid_words = [
            # High-frequency starting words
            "AROSE", "ADIEU", "AUDIO", "OUNCE", "MEDIA", "TRAIN", "SLATE", "CRANE",
            "IRATE", "TRACE", "CRATE", "STARE", "RAISE", "ARISE", "LEARN", "TEARS",
            "HEART", "STONE", "HOUSE", "MOUSE", "ABOUT", "AFTER", "AGAIN", "ABOVE",
            "ABUSE", "ACTOR", "ACUTE", "ADMIT", "ADOPT", "ADULT", "AGENT", "AGREE",
            "AHEAD", "ALARM", "ALBUM", "ALERT", "ALIEN", "ALIGN", "ALIKE", "ALIVE",
            "ALLOW", "ALONE", "ALONG", "ALTER", "AMONG", "ANGEL", "ANGER", "ANGLE",
            "ANGRY", "APART", "APPLE", "APPLY", "ARENA", "ARGUE", "ARISE", "ARRAY",
            "ASIDE", "ASSET", "AVOID", "AWAKE", "AWARD", "AWARE", "BADLY", "BAKER",
            "BASES", "BASIC", "BEACH", "BEGAN", "BEGIN", "BEING", "BELOW", "BENCH",
            "BILLY", "BIRTH", "BLACK", "BLAME", "BLANK", "BLIND", "BLOCK", "BLOOD",
            "BOARD", "BOOST", "BOOTH", "BOUND", "BRAIN", "BRAND", "BRASS", "BRAVE",
            "BREAD", "BREAK", "BREED", "BRIEF", "BRING", "BROAD", "BROKE", "BROWN",
            "BUILD", "BUILT", "BUYER", "CABLE", "CALIF", "CARRY", "CATCH", "CAUSE",
            "CHAIN", "CHAIR", "CHAOS", "CHARM", "CHART", "CHASE", "CHEAP", "CHECK",
            "CHEST", "CHIEF", "CHILD", "CHINA", "CHOSE", "CIVIL", "CLAIM", "CLASS",
            "CLEAN", "CLEAR", "CLICK", "CLIMB", "CLOCK", "CLOSE", "CLOUD", "COACH",
            "COAST", "COULD", "COUNT", "COURT", "COVER", "CRAFT", "CRASH", "CRAZY",
            "CREAM", "CRIME", "CROSS", "CROWD", "CROWN", "CRUDE", "CURVE", "CYCLE",
            "DAILY", "DANCE", "DATED", "DEALT", "DEATH", "DEBUT", "DELAY", "DEPTH",
            "DOING", "DOUBT", "DOZEN", "DRAFT", "DRAMA", "DRANK", "DRAWN", "DREAM",
            "DRESS", "DRILL", "DRINK", "DRIVE", "DROVE", "DYING", "EAGER", "EARLY",
            "EARTH", "EIGHT", "ELITE", "EMPTY", "ENEMY", "ENJOY", "ENTER", "ENTRY",
            "EQUAL", "ERROR", "EVENT", "EVERY", "EXACT", "EXIST", "EXTRA", "FAITH",
            "FALSE", "FAULT", "FIBER", "FIELD", "FIFTH", "FIFTY", "FIGHT", "FINAL",
            "FIRST", "FIXED", "FLASH", "FLEET", "FLOOR", "FLUID", "FOCUS", "FORCE",
            "FORTH", "FORTY", "FORUM", "FOUND", "FRAME", "FRANK", "FRAUD", "FRESH",
            "FRONT", "FRUIT", "FULLY", "FUNNY", "GIANT", "GIVEN", "GLASS", "GLOBE",
            "GOING", "GRACE", "GRADE", "GRAIN", "GRAND", "GRANT", "GRASS", "GRAVE",
            "GREAT", "GREEN", "GROSS", "GROUP", "GROWN", "GUARD", "GUESS", "GUEST",
            "GUIDE", "HAPPY", "HARRY", "HEART", "HEAVY", "HENCE", "HENRY", "HORSE",
            "HOTEL", "HOUSE", "HUMAN", "IDEAL", "IMAGE", "INDEX", "INNER", "INPUT",
            "ISSUE", "JAPAN", "JIMMY", "JOINT", "JONES", "JUDGE", "KNOWN", "LABEL",
            "LARGE", "LASER", "LATER", "LAUGH", "LAYER", "LEARN", "LEASE", "LEAST",
            "LEAVE", "LEGAL", "LEVEL", "LEWIS", "LIGHT", "LIMIT", "LINKS", "LIVES",
            "LOCAL", "LOOSE", "LOWER", "LUCKY", "LUNCH", "LYING", "MAGIC", "MAJOR",
            "MAKER", "MARCH", "MARIA", "MATCH", "MAYBE", "MAYOR", "MEANT", "MEDIA",
            "METAL", "MIGHT", "MINOR", "MINUS", "MIXED", "MODEL", "MONEY", "MONTH",
            "MORAL", "MOTOR", "MOUNT", "MOUSE", "MOUTH", "MOVED", "MOVIE", "MUSIC",
            "NEEDS", "NEVER", "NEWLY", "NIGHT", "NOISE", "NORTH", "NOTED", "NOVEL",
            "NURSE", "OCCUR", "OCEAN", "OFFER", "OFTEN", "ORDER", "OTHER", "OUGHT",
            "PAINT", "PANEL", "PAPER", "PARTY", "PEACE", "PETER", "PHASE", "PHONE",
            "PHOTO", "PIANO", "PIECE", "PILOT", "PITCH", "PLACE", "PLAIN", "PLANE",
            "PLANT", "PLATE", "POINT", "POUND", "POWER", "PRESS", "PRICE", "PRIDE",
            "PRIME", "PRINT", "PRIOR", "PRIZE", "PROOF", "PROUD", "PROVE", "QUEEN",
            "QUICK", "QUIET", "QUITE", "RADIO", "RAISE", "RANGE", "RAPID", "RATIO",
            "REACH", "READY", "REALM", "REBEL", "REFER", "RELAX", "RELAY", "REPLY",
            "RIGHT", "RIVAL", "RIVER", "ROBIN", "ROGER", "ROMAN", "ROUGH", "ROUND",
            "ROUTE", "ROYAL", "RURAL", "SCALE", "SCENE", "SCOPE", "SCORE", "SENSE",
            "SERVE", "SEVEN", "SHALL", "SHAPE", "SHARE", "SHARP", "SHEET", "SHELF",
            "SHELL", "SHIFT", "SHINE", "SHIRT", "SHOCK", "SHOOT", "SHORT", "SHOWN",
            "SIDES", "SIGHT", "SILLY", "SINCE", "SIXTH", "SIXTY", "SIZED", "SKILL",
            "SLEEP", "SLIDE", "SMALL", "SMART", "SMILE", "SMITH", "SMOKE", "SNAKE",
            "SNOW", "SOLID", "SOLVE", "SORRY", "SOUND", "SOUTH", "SPACE", "SPARE",
            "SPEAK", "SPEED", "SPEND", "SPENT", "SPLIT", "SPOKE", "SPORT", "SQUAD",
            "STAFF", "STAGE", "STAKE", "STAND", "START", "STATE", "STEAM", "STEEL",
            "STEEP", "STEER", "STICK", "STILL", "STOCK", "STONE", "STOOD", "STORE",
            "STORM", "STORY", "STRIP", "STUCK", "STUDY", "STUFF", "STYLE", "SUGAR",
            "SUITE", "SUPER", "SWEET", "TABLE", "TAKEN", "TASTE", "TAXES", "TEACH",
            "TEAMS", "TEETH", "TERRY", "TEXAS", "THANK", "THEFT", "THEIR", "THEME",
            "THERE", "THESE", "THICK", "THING", "THINK", "THIRD", "THOSE", "THREE",
            "THREW", "THROW", "THUMB", "TIGHT", "TIMES", "TIRED", "TITLE", "TODAY",
            "TOPIC", "TOTAL", "TOUCH", "TOUGH", "TOWER", "TRACK", "TRADE", "TRAIN",
            "TRAIT", "TRASH", "TREAT", "TREND", "TRIAL", "TRIBE", "TRICK", "TRIED",
            "TRIES", "TRUCK", "TRULY", "TRUNK", "TRUST", "TRUTH", "TWICE", "TWIST",
            "TYLER", "UNCLE", "UNDER", "UNDUE", "UNION", "UNITY", "UNTIL", "UPPER",
            "UPSET", "URBAN", "USAGE", "USUAL", "VALID", "VALUE", "VIDEO", "VIRUS",
            "VISIT", "VITAL", "VOCAL", "VOICE", "WASTE", "WATCH", "WATER", "WHEEL",
            "WHERE", "WHICH", "WHILE", "WHITE", "WHOLE", "WHOSE", "WOMAN", "WOMEN",
            "WORLD", "WORRY", "WORSE", "WORST", "WORTH", "WOULD", "WRITE", "WRONG",
            "WROTE", "YOUNG", "YOUTH"
        ]

        # Answer words (subset of valid words that can be answers)
        answer_words = [
            "AROSE", "ADIEU", "AUDIO", "OUNCE", "MEDIA", "TRAIN", "SLATE", "CRANE",
            "IRATE", "TRACE", "CRATE", "STARE", "RAISE", "ARISE", "LEARN", "TEARS",
            "HEART", "STONE", "HOUSE", "MOUSE", "ABOUT", "AFTER", "AGAIN", "ABOVE",
            "ABUSE", "ACTOR", "ACUTE", "ADMIT", "ADOPT", "ADULT", "AGENT", "AGREE",
            "AHEAD", "ALARM", "ALBUM", "ALERT", "ALIEN", "ALIGN", "ALIKE", "ALIVE",
            "ALLOW", "ALONE", "ALONG", "ALTER", "AMONG", "ANGEL", "ANGER", "ANGLE",
            "ANGRY", "APART", "APPLE", "APPLY", "ARENA", "ARGUE", "ARRAY", "ASIDE",
            "ASSET", "AVOID", "AWAKE", "AWARD", "AWARE", "BADLY", "BAKER", "BASES",
            "BASIC", "BEACH", "BEGAN", "BEGIN", "BEING", "BELOW", "BENCH", "BILLY",
            "BIRTH", "BLACK", "BLAME", "BLANK", "BLIND", "BLOCK", "BLOOD", "BOARD",
            "BOOST", "BOOTH", "BOUND", "BRAIN", "BRAND", "BRASS", "BRAVE", "BREAD",
            "BREAK", "BREED", "BRIEF", "BRING", "BROAD", "BROKE", "BROWN", "BUILD",
            "BUILT", "BUYER", "CABLE", "CARRY", "CATCH", "CAUSE", "CHAIN", "CHAIR",
            "CHAOS", "CHARM", "CHART", "CHASE", "CHEAP", "CHECK", "CHEST", "CHIEF",
            "CHILD", "CHINA", "CHOSE", "CIVIL", "CLAIM", "CLASS", "CLEAN", "CLEAR",
            "CLICK", "CLIMB", "CLOCK", "CLOSE", "CLOUD", "COACH", "COAST", "COULD",
            "COUNT", "COURT", "COVER", "CRAFT", "CRASH", "CRAZY", "CREAM", "CRIME",
            "CROSS", "CROWD", "CROWN", "CRUDE", "CURVE", "CYCLE", "DAILY", "DANCE",
            "DATED", "DEALT", "DEATH", "DEBUT", "DELAY", "DEPTH", "DOING", "DOUBT",
            "DOZEN", "DRAFT", "DRAMA", "DRANK", "DRAWN", "DREAM", "DRESS", "DRILL",
            "DRINK", "DRIVE", "DROVE", "DYING", "EAGER", "EARLY", "EARTH", "EIGHT",
            "ELITE", "EMPTY", "ENEMY", "ENJOY", "ENTER", "ENTRY", "EQUAL", "ERROR",
            "EVENT", "EVERY", "EXACT", "EXIST", "EXTRA", "FAITH", "FALSE", "FAULT",
            "FIBER", "FIELD", "FIFTH", "FIFTY", "FIGHT", "FINAL", "FIRST", "FIXED",
            "FLASH", "FLEET", "FLOOR", "FLUID", "FOCUS", "FORCE", "FORTH", "FORTY",
            "FORUM", "FOUND", "FRAME", "FRANK", "FRAUD", "FRESH", "FRONT", "FRUIT",
            "FULLY", "FUNNY", "GIANT", "GIVEN", "GLASS", "GLOBE", "GOING", "GRACE",
            "GRADE", "GRAIN", "GRAND", "GRANT", "GRASS", "GRAVE", "GREAT", "GREEN",
            "GROSS", "GROUP", "GROWN", "GUARD", "GUESS", "GUEST", "GUIDE", "HAPPY",
            "HEART", "HEAVY", "HENCE", "HORSE", "HOTEL", "HOUSE", "HUMAN", "IDEAL",
            "IMAGE", "INDEX", "INNER", "INPUT", "ISSUE", "JOINT", "JUDGE", "KNOWN",
            "LABEL", "LARGE", "LASER", "LATER", "LAUGH", "LAYER", "LEARN", "LEASE",
            "LEAST", "LEAVE", "LEGAL", "LEVEL", "LIGHT", "LIMIT", "LINKS", "LIVES",
            "LOCAL", "LOOSE", "LOWER", "LUCKY", "LUNCH", "LYING", "MAGIC", "MAJOR",
            "MAKER", "MARCH", "MATCH", "MAYBE", "MAYOR", "MEANT", "MEDIA", "METAL",
            "MIGHT", "MINOR", "MINUS", "MIXED", "MODEL", "MONEY", "MONTH", "MORAL",
            "MOTOR", "MOUNT", "MOUSE", "MOUTH", "MOVED", "MOVIE", "MUSIC", "NEEDS",
            "NEVER", "NEWLY", "NIGHT", "NOISE", "NORTH", "NOTED", "NOVEL", "NURSE",
            "OCCUR", "OCEAN", "OFFER", "OFTEN", "ORDER", "OTHER", "OUGHT", "PAINT",
            "PANEL", "PAPER", "PARTY", "PEACE", "PHASE", "PHONE", "PHOTO", "PIANO",
            "PIECE", "PILOT", "PITCH", "PLACE", "PLAIN", "PLANE", "PLANT", "PLATE",
            "POINT", "POUND", "POWER", "PRESS", "PRICE", "PRIDE", "PRIME", "PRINT",
            "PRIOR", "PRIZE", "PROOF", "PROUD", "PROVE", "QUEEN", "QUICK", "QUIET",
            "QUITE", "RADIO", "RAISE", "RANGE", "RAPID", "RATIO", "REACH", "READY",
            "REALM", "REBEL", "REFER", "RELAX", "RELAY", "REPLY", "RIGHT", "RIVAL",
            "RIVER", "ROUGH", "ROUND", "ROUTE", "ROYAL", "RURAL", "SCALE", "SCENE",
            "SCOPE", "SCORE", "SENSE", "SERVE", "SEVEN", "SHALL", "SHAPE", "SHARE",
            "SHARP", "SHEET", "SHELF", "SHELL", "SHIFT", "SHINE", "SHIRT", "SHOCK",
            "SHOOT", "SHORT", "SHOWN", "SIDES", "SIGHT", "SILLY", "SINCE", "SIXTH",
            "SIXTY", "SIZED", "SKILL", "SLEEP", "SLIDE", "SMALL", "SMART", "SMILE",
            "SMOKE", "SOLID", "SOLVE", "SORRY", "SOUND", "SOUTH", "SPACE", "SPARE",
            "SPEAK", "SPEED", "SPEND", "SPENT", "SPLIT", "SPOKE", "SPORT", "SQUAD",
            "STAFF", "STAGE", "STAKE", "STAND", "START", "STATE", "STEAM", "STEEL",
            "STEEP", "STEER", "STICK", "STILL", "STOCK", "STONE", "STOOD", "STORE",
            "STORM", "STORY", "STRIP", "STUCK", "STUDY", "STUFF", "STYLE", "SUGAR",
            "SUITE", "SUPER", "SWEET", "TABLE", "TAKEN", "TASTE", "TAXES", "TEACH",
            "TEAMS", "TEETH", "THANK", "THEFT", "THEIR", "THEME", "THERE", "THESE",
            "THICK", "THING", "THINK", "THIRD", "THOSE", "THREE", "THREW", "THROW",
            "THUMB", "TIGHT", "TIMES", "TIRED", "TITLE", "TODAY", "TOPIC", "TOTAL",
            "TOUCH", "TOUGH", "TOWER", "TRACK", "TRADE", "TRAIN", "TRAIT", "TRASH",
            "TREAT", "TREND", "TRIAL", "TRIBE", "TRICK", "TRIED", "TRIES", "TRUCK",
            "TRULY", "TRUNK", "TRUST", "TRUTH", "TWICE", "TWIST", "UNCLE", "UNDER",
            "UNION", "UNITY", "UNTIL", "UPPER", "UPSET", "URBAN", "USAGE", "USUAL",
            "VALID", "VALUE", "VIDEO", "VIRUS", "VISIT", "VITAL", "VOCAL", "VOICE",
            "WASTE", "WATCH", "WATER", "WHEEL", "WHERE", "WHICH", "WHILE", "WHITE",
            "WHOLE", "WHOSE", "WOMAN", "WOMEN", "WORLD", "WORRY", "WORSE", "WORST",
            "WORTH", "WOULD", "WRITE", "WRONG", "WROTE", "YOUNG", "YOUTH"
        ]

        self._valid_words = set(valid_words)
        self._answer_words = set(answer_words)

    def get_word_stats(self) -> dict[str, int]:
        """Get statistics about loaded word lists.

        Returns:
            Dictionary with word list statistics
        """
        return {
            "total_valid_words": len(self._valid_words),
            "total_answer_words": len(self._answer_words),
            "answer_percentage": (len(self._answer_words) / len(self._valid_words) * 100) if self._valid_words else 0
        }
