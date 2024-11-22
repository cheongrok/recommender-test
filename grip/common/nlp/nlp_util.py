# nlp_util.py
import re

# Emoji ranges categorized for better readability
EMOJI_RANGES = [
    "\U0001f1e0-\U0001f1ff",  # Flags (iOS)
    "\U0001f300-\U0001f5ff",  # Symbols & Pictographs
    "\U0001f600-\U0001f64f",  # Emoticons
    "\U0001f680-\U0001f6ff",  # Transport & Map Symbols
    "\U0001f700-\U0001f77f",  # Alchemical Symbols
    "\U0001f780-\U0001f7ff",  # Geometric Shapes Extended
    "\U0001f800-\U0001f8ff",  # Supplemental Arrows-C
    "\U0001f900-\U0001f9ff",  # Supplemental Symbols and Pictographs
    "\U0001fa00-\U0001fa6f",  # Chess Symbols
    "\U0001fa70-\U0001faff",  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027b0",  # Dingbats
]
EMOJI_PATTERN = f"{''.join(EMOJI_RANGES)}"
EMOJI_REGEX = re.compile(f"[{EMOJI_PATTERN}]")
REPEATED_CHARS_REGEX = re.compile(f"([{EMOJI_PATTERN}\w])\\1{{2,}}")
PRICE_REGEX = re.compile("￦?[0-9,]{4,}원?")
MULTI_SPACE_REGEX = re.compile("\s+")


def normalize_repeated_characters(input_text, num_repeats=2):
    """Normalize repeated characters and emojis in a sentence to a specified number of repetitions."""
    if num_repeats > 0:
        input_text = REPEATED_CHARS_REGEX.sub(r"\1" * num_repeats, input_text)
    input_text = MULTI_SPACE_REGEX.sub(" ", input_text)
    return input_text.strip()


def remove_emojis(input_text):
    """Remove emojis from a sentence to leave only text."""
    return EMOJI_REGEX.sub(" ", input_text)


def remove_prices(input_text):
    """Remove price patterns from a sentence, cleaning up any financial figures."""
    return PRICE_REGEX.sub(" ", input_text)
