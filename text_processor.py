"""
Text Processing Module

This module provides a TextProcessor class for cleaning and preprocessing text data,
with special handling for financial text. It includes methods for:
- Cleaning text by removing URLs, emojis, special characters, and numbers
- Extracting stock tickers from text using POS tagging and pattern matching
- Tokenizing text into words
- Removing stop words
- Lemmatizing words

The TextProcessor class is designed to work with financial text, particularly
from social media platforms like Reddit, and includes special handling for
stock tickers and financial terminology.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import string # Import string for punctuation removal during POS tagging preparation

# --- Download required NLTK data (ensure these run successfully) ---
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    # Continue without the data - the application will handle missing resources gracefully
# ------------------------------------------------------------------

class TextProcessor:
    """
    A class for processing text data, with special handling for financial text.

    This class provides methods for cleaning and preprocessing text data,
    with a focus on financial text from social media platforms. It includes
    special handling for stock tickers and financial terminology using POS tagging
    and refined pattern matching.

    Attributes:
        stop_words (set): Base set of stop words from NLTK.
        lemmatizer (WordNetLemmatizer): Lemmatizer for reducing words to base form.
        url_pattern (re.Pattern): Regular expression for matching URLs.
        emoji_pattern (re.Pattern): Regular expression for matching emojis.
        # Patterns for ticker extraction
        cashtag_pattern (re.Pattern): Finds $TICKER patterns.
        ticker_format_pattern (re.Pattern): Checks if a word matches ticker format (e.g., ABC, XY.Z).
        # Expanded list to exclude common words/slang mistaken for tickers
        excluded_words (set): Lowercase common English words, WSB slang, and trading terms.
    """

    def __init__(self):
        """
        Initialize the text processor with stop words, lemmatizer, regex patterns,
        and an expanded exclusion list.
        """
        # Basic NLTK setup
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # --- Regular expressions ---
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        # Keep $ for initial cashtag finding, remove other special chars later
        self.initial_special_char_pattern = re.compile(r'[^\w\s$]') # Allow word chars, whitespace, $
        self.final_special_char_pattern = re.compile(r'[^\w\s]') # Allow only word chars and whitespace
        self.number_pattern = re.compile(r'\b\d+\b') # Matches standalone numbers

        # --- Ticker patterns ---
        # Pattern for cashtags: $ followed by 1-5 letters, optional . then 1-2 letters
        self.cashtag_pattern = re.compile(r'\$[A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?\b')
        # Pattern to check if a word *looks* like a ticker (used after POS tagging)
        # 1-5 letters, optionally followed by . and 1-2 letters. Anchored start/end.
        self.ticker_format_pattern = re.compile(r'^[A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?$')
        # Simple crypto pattern (can be expanded or made more complex)
        # self.crypto_pattern = re.compile(r'\b(BTC|ETH|DOGE|XRP|ADA|SOL|DOT|LINK|UNI|AAVE)\b', re.IGNORECASE) # Example

        # --- Exclusion List ---
        # Combine NLTK stop words with common English words, WSB slang, financial terms, etc.
        # All words should be lowercase.
        self.excluded_words = set(stopwords.words('english')) | {
            # Common short words often capitalized
            'a', 'i', 'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'hi', 'if', 'in', 'is', 'it',
            'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we',
            # Common longer words
            'ceo', 'cto', 'cfo', 'coo', 'sec', 'fed', 'irs', 'nyse', 'nasdaq', 'dow',
            'news', 'data', 'info', 'link', 'post', 'pdf', 'html', ' P ', ' D ',
            'inc', 'corp', 'llc', 'ltd', 'etf', 'adr', 'ipo', 'spac', 'eps', 'ev',
            'buy', 'sell', 'hold', 'call', 'put', 'long', 'short', 'stock', 'share', 'trade',
            'market', 'money', 'cash', 'gain', 'loss', 'profit', 'risk', 'fund', 'hedge',
            'play', 'move', 'moon', 'ape', 'yolo', 'fomo', 'hodl', 'dd', 'ath', 'bag',
            'tendies', 'diamond', 'hands', 'rocket', 'squeeze', 'bear', 'bull', 'gay',
            'bot', 'mod', 'wsb', 'sub', 'reddit', 'mods', 'admin', 'alert', 'baby',
            'beta', 'big', 'bro', 'chart', 'cool', 'cost', 'date', 'days', 'down', 'dude',
            'earn', 'else', 'even', 'ever', 'feel', 'file', 'find', 'fine', 'free',
            'from', 'full', 'game', 'give', 'good', 'got', 'guy', 'guys', 'have', 'help',
            'here', 'high', 'hour', 'idea', 'into', 'join', 'just', 'keep', 'know', 'last',
            'late', 'left', 'less', 'life', 'like', 'line', 'list', 'live', 'look', 'lot',
            'love', 'low', 'made', 'main', 'make', 'many', 'meme', ' P ', ' D ',
            'min', 'mins', 'more', 'most', 'much', 'must', 'name', 'need', 'nice',
            'nope', 'note', 'now', 'open', 'part', 'past', 'pay', 'plan', 'pls', 'plz',
            'pm', 'pdt', 'est', 'edt', 'gmt', 'pst',
            'point', 'psa', 'real', 'red', 'read', 'ride', 'righ', 'room', 'run', 'said',
            'same', 'save', 'say', 'see', 'seem', 'sent', 'set', 'shit', 'show', 'sick',
            'side', 'sign', 'soon', 'sort', 'spot', 'star', 'stay', 'step', 'stop',
            'sure', 'talk', 'tell', 'term', 'test', 'text', 'than', 'that', 'them',
            'then', 'they', 'this', 'tho', ' F ',
            'thus', 'till', 'time', 'tip', 'told', 'took', 'top', 'true', 'try', 'turn',
            'usd', 'use', 'used', 'user', 'view', 'wait', 'want', 'week', 'well', 'went',
            'were', 'what', 'when', 'who', 'why', 'will', 'win', 'with', 'word', 'work',
            'year', 'yep', 'yes', 'yet', 'you', 'your', 'utc',
            # Single letters that might be common or NLTK artifacts
            'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        }


    def clean_text(self, text, stage='final'):
        """
        Clean text by removing URLs, emojis, special characters, and numbers.
        Stage 'initial' preserves $, 'final' removes all non-alphanumeric.

        Args:
            text (str): Text to clean.
            stage (str): 'initial' for POS tagging prep, 'final' for analysis prep.

        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = self.url_pattern.sub(' ', text)
        text = self.emoji_pattern.sub(' ', text)

        if stage == 'initial':
            # Keep $ for cashtag finding, remove other punctuation
            text = self.initial_special_char_pattern.sub(' ', text)
        else: # stage == 'final'
            # Remove all special chars including $ after ticker extraction
             text = self.final_special_char_pattern.sub(' ', text)

        # Remove standalone numbers - careful not to remove numbers within potential future tickers if pattern changes
        text = self.number_pattern.sub(' ', text)
        text = ' '.join(text.split()) # Normalize whitespace
        return text

    def extract_tickers(self, text):
        """
        Extract potential stock tickers using cashtags and POS-tag guided filtering.

        1. Finds high-confidence cashtags ($XYZ).
        2. Tokenizes and POS tags the original text.
        3. Identifies Proper Nouns (NNP/NNPS) that match ticker format.
        4. Filters these candidates against an exclusion list.
        5. Combines cashtags and filtered standalone tickers.

        Args:
            text (str): The original text to extract tickers from.

        Returns:
            list: Sorted list of unique potential tickers found (uppercase).
        """
        if not isinstance(text, str):
            return []

        # 1. Find Cashtags (High Confidence)
        # Clean slightly for cashtag finding (remove URLs, emojis, minimal punctuation)
        initial_cleaned_text = self.clean_text(text, stage='initial')
        cashtags = self.cashtag_pattern.findall(initial_cleaned_text)
        # Remove '$' and convert to uppercase
        confirmed_tickers = {tag[1:].upper() for tag in cashtags if self.ticker_format_pattern.match(tag[1:])}

        # 2. Tokenize and POS Tag the *original* text for context
        # Minimal cleaning for tokenization: remove URLs, emojis
        pos_text = self.url_pattern.sub(' ', text)
        pos_text = self.emoji_pattern.sub(' ', pos_text)
        tokens = word_tokenize(pos_text)
        tagged_tokens = pos_tag(tokens)

        # 3. Identify potential standalone tickers from Proper Nouns (NNP, NNPS)
        potential_standalone = set()
        for word, tag in tagged_tokens:
            # Check if it's a Proper Noun and looks like a ticker
            if tag in ('NNP', 'NNPS') and self.ticker_format_pattern.match(word):
                # Check if it's not an excluded word (case-insensitive check)
                if word.lower() not in self.excluded_words:
                    potential_standalone.add(word.upper())

        # 4. Combine cashtags and validated standalone tickers
        all_potential_tickers = confirmed_tickers | potential_standalone

        # 5. Final filtering: Ensure confirmed tickers aren't excluded words
        # (This is slightly redundant if cashtags logic is robust, but safe)
        final_tickers = {ticker for ticker in all_potential_tickers if ticker.lower() not in self.excluded_words}


        # Optional: Add explicit crypto check if needed
        # crypto_tickers = set(self.crypto_pattern.findall(text.upper()))
        # final_tickers |= crypto_tickers # Add crypto if using separate pattern

        return sorted(list(final_tickers))

    def tokenize(self, text):
        """Tokenize text into words after final cleaning."""
        # Use final cleaning stage before tokenizing for analysis
        cleaned_text = self.clean_text(text, stage='final')
        return word_tokenize(cleaned_text)

    def remove_stop_words(self, tokens, identified_tickers):
        """
        Remove stop words, preserving identified tickers.

        Args:
            tokens (list): List of tokens from tokenize().
            identified_tickers (set): Set of tickers identified by extract_tickers().

        Returns:
            list: Filtered list of tokens.
        """
        identified_tickers_lower = {t.lower() for t in identified_tickers}
        # Use the main exclusion list (which includes NLTK stopwords)
        return [token for token in tokens if token not in self.excluded_words or token in identified_tickers_lower]


    def get_wordnet_pos(self, treebank_tag):
        """Map Treebank POS tags to WordNet POS tags."""
        if treebank_tag.startswith('J'):
            return 'a'  # adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # verb
        elif treebank_tag.startswith('N'):
            return 'n'  # noun
        elif treebank_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun

    def lemmatize(self, tokens, identified_tickers):
        """
        Lemmatize tokens using WordNet, preserving identified tickers.

        Args:
            tokens (list): List of tokens (ideally after stop word removal).
            identified_tickers (set): Set of tickers identified by extract_tickers().

        Returns:
            list: Lemmatized list of tokens.
        """
        identified_tickers_lower = {t.lower() for t in identified_tickers}
        lemmatized_tokens = []
        tagged_tokens = pos_tag(tokens) # POS tag the filtered tokens

        for token, tag in tagged_tokens:
            # Preserve identified tickers (case-insensitive check)
            if token in identified_tickers_lower:
                lemmatized_tokens.append(token.upper()) # Keep tickers uppercase
            else:
                wordnet_pos = self.get_wordnet_pos(tag)
                lemma = self.lemmatizer.lemmatize(token, pos=wordnet_pos)
                lemmatized_tokens.append(lemma)
        return lemmatized_tokens

    def process_text(self, text):
        """
        Full processing pipeline: Extract tickers, clean, tokenize, remove stops, lemmatize.

        Args:
            text (str): Raw text input.

        Returns:
            tuple: (list of identified tickers, list of processed tokens)
        """
        # 1. Extract tickers first from original text
        identified_tickers = self.extract_tickers(text)
        identified_tickers_set = set(identified_tickers) # Use set for faster lookups

        # 2. Clean text for analysis (final stage)
        cleaned_text = self.clean_text(text, stage='final')

        # 3. Tokenize the finally cleaned text
        tokens = word_tokenize(cleaned_text) # Already lowercase

        # 4. Remove stop words, preserving identified tickers
        tokens = self.remove_stop_words(tokens, identified_tickers_set)

        # 5. Lemmatize, preserving identified tickers
        processed_tokens = self.lemmatize(tokens, identified_tickers_set)

        return identified_tickers, processed_tokens