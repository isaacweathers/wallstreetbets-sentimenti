"""
Text Processing Module

This module provides a TextProcessor class for cleaning and preprocessing text data,
with special handling for financial text. It includes methods for:
- Cleaning text by removing URLs, emojis, special characters, and numbers
- Extracting stock tickers from text
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
import logging

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    A class for processing text data, with special handling for financial text.
    
    This class provides methods for cleaning and preprocessing text data,
    with a focus on financial text from social media platforms. It includes
    special handling for stock tickers and financial terminology.
    
    Attributes:
        stop_words (set): Set of stop words to remove from text
        lemmatizer (WordNetLemmatizer): Lemmatizer for reducing words to base form
        financial_stop_words (set): Set of financial-specific stop words
        url_pattern (re.Pattern): Regular expression for matching URLs
        emoji_pattern (re.Pattern): Regular expression for matching emojis
        special_char_pattern (re.Pattern): Regular expression for matching special characters
        number_pattern (re.Pattern): Regular expression for matching numbers
        cashtag_pattern (re.Pattern): Regular expression for matching cashtags ($TICKER)
        ticker_pattern (re.Pattern): Regular expression for matching tickers
        crypto_pattern (re.Pattern): Regular expression for matching cryptocurrency tickers
        common_words (set): Set of common words that might be mistaken for tickers
    """
    
    def __init__(self):
        """
        Initialize the text processor with stop words, lemmatizer, and regular expressions.
        """
        # Initialize stop words from NLTK
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Custom stop words for financial context
        self.financial_stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might',
            'must', 'can', 'could', 'this', 'that', 'these', 'those', 'my', 'your',
            'his', 'her', 'its', 'our', 'their', 'am', 'pm', 'et', 'al', 'etc',
            'vs', 'vs.', 'e.g', 'e.g.', 'i.e', 'i.e.', 'etc.', 'etc', '&', 'and',
            'or', 'if', 'then', 'else', 'when', 'where', 'why', 'how', 'what',
            'who', 'which', 'whom', 'whose', 'there', 'here', 'now', 'then',
            'today', 'tomorrow', 'yesterday', 'week', 'month', 'year', 'time',
            'day', 'night', 'morning', 'evening', 'afternoon', 'monday', 'tuesday',
            'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'january',
            'february', 'march', 'april', 'may', 'june', 'july', 'august',
            'september', 'october', 'november', 'december'
        }
        
        # Add financial stop words to the standard stop words
        self.stop_words.update(self.financial_stop_words)
        
        # Regular expressions for cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        self.special_char_pattern = re.compile(r'[^a-zA-Z0-9\s$]')
        self.number_pattern = re.compile(r'\b\d+\b')
        
        # Ticker patterns
        self.cashtag_pattern = re.compile(r'\$[A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?')
        self.ticker_pattern = re.compile(r'(?:^|\s)([A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?)(?:\s|$)')
        self.crypto_pattern = re.compile(r'(?:^|\s)(BTC|ETH|DOGE|XRP|ADA|SOL|DOT|LINK|UNI|AAVE)(?:\s|$)')
        
        # Common words that might be mistaken for tickers
        self.common_words = {
            'a', 'i', 'am', 'an', 'at', 'by', 'do', 'go', 'hi', 'if', 'in', 'it', 'me', 'my', 'no', 'of', 
            'on', 'or', 'so', 'to', 'up', 'us', 'we', 'he', 'she', 'the', 'and', 'for', 'are', 'but', 
            'not', 'you', 'all', 'any', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 
            'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'with', 'have', 'this', 
            'will', 'your', 'from', 'they', 'know', 'people', 'into', 'year', 'good', 'some', 'could', 
            'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 
            'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 
            'way', ' ',
        }
        
    def clean_text(self, text):
        """
        Clean the text by removing URLs, emojis, special characters, and numbers.
        Preserve $ symbols for tickers.
        
        This method:
        1. Converts text to lowercase
        2. Removes URLs
        3. Removes emojis
        4. Removes special characters (except $ for tickers)
        5. Removes standalone numbers
        6. Removes extra whitespace
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove emojis
        text = self.emoji_pattern.sub(' ', text)
        
        # Remove special characters (keep $ for tickers)
        text = self.special_char_pattern.sub(' ', text)
        
        # Remove standalone numbers (but keep numbers that are part of tickers)
        text = self.number_pattern.sub('', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def is_ticker(self, word):
        """
        Check if a word is a stock ticker.
        Tickers are typically 1-5 characters long and may contain $.
        
        This method:
        1. Removes $ if present
        2. Checks if the word matches the ticker pattern
        3. Checks if the word is not in the common words list
        
        Args:
            word (str): Word to check
            
        Returns:
            bool: True if the word is a ticker, False otherwise
        """
        # Remove $ if present
        word = word.replace('$', '')
        
        # Check if it's a valid ticker format
        return bool(re.match(r'^[A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?$', word)) and word.upper() not in self.common_words
    
    def extract_tickers(self, text):
        """
        Extract stock tickers from text.
        Returns a list of unique tickers found in the text.
        
        This method:
        1. Finds cashtags ($TICKER)
        2. Finds standalone tickers (without $)
        3. Finds crypto tickers
        4. Filters out common words that might be mistaken for tickers
        5. Removes duplicates and sorts
        
        Args:
            text (str): Text to extract tickers from
            
        Returns:
            list: List of unique tickers found in the text
        """
        if not isinstance(text, str):
            return []
            
        # Find cashtags ($TICKER)
        cashtags = self.cashtag_pattern.findall(text)
        cashtags = [tag.replace('$', '') for tag in cashtags]
        
        # Find standalone tickers (without $)
        standalone_tickers = self.ticker_pattern.findall(text)
        
        # Find crypto tickers
        crypto_tickers = self.crypto_pattern.findall(text)
        
        # Combine all tickers
        all_tickers = cashtags + standalone_tickers + crypto_tickers
        
        # Filter out common words that might be mistaken for tickers
        tickers = [ticker.upper() for ticker in all_tickers if ticker.upper() not in self.common_words]
        
        # Remove duplicates and sort
        return sorted(list(set(tickers)))
    
    def tokenize(self, text):
        """
        Tokenize the text into words.
        
        This method:
        1. Uses NLTK's word_tokenize to split text into words
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens (words)
        """
        return word_tokenize(text)
    
    def remove_stop_words(self, tokens):
        """
        Remove stop words from the token list.
        Preserve tickers even if they match stop words.
        
        This method:
        1. Filters out tokens that are in the stop words list
        2. Preserves tokens that are tickers, even if they match stop words
        
        Args:
            tokens (list): List of tokens to filter
            
        Returns:
            list: Filtered list of tokens
        """
        return [token for token in tokens if token not in self.stop_words or self.is_ticker(token)]
    
    def lemmatize(self, tokens):
        """
        Lemmatize tokens using WordNet lemmatizer with part-of-speech tagging.
        
        Args:
            tokens (list): List of tokens to lemmatize
            
        Returns:
            list: List of lemmatized tokens
        """
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = []
        
        for token in tokens:
            try:
                # Try to get part-of-speech tag
                pos_tagged = pos_tag([token])[0][1]
                
                # Map NLTK POS tags to WordNet POS tags
                pos = self._get_wordnet_pos(pos_tagged)
                
                # Lemmatize with POS tag
                lemmatized = lemmatizer.lemmatize(token, pos=pos)
                lemmatized_tokens.append(lemmatized)
            except (LookupError, IndexError) as e:
                # If tagger is not available or tagging fails, use default lemmatization
                logger.warning(f"POS tagging failed for token '{token}': {e}. Using default lemmatization.")
                lemmatized = lemmatizer.lemmatize(token)
                lemmatized_tokens.append(lemmatized)
        
        return lemmatized_tokens
    
    def process_text(self, text):
        """
        Process the text through all cleaning and preprocessing steps.
        
        This method:
        1. Cleans the text
        2. Tokenizes the text
        3. Removes stop words
        4. Lemmatizes the tokens
        
        Args:
            text (str): Text to process
            
        Returns:
            list: Processed list of tokens
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stop words
        tokens = self.remove_stop_words(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        return tokens
    
    def _get_wordnet_pos(self, pos_tag):
        """
        Map NLTK POS tags to WordNet POS tags.
        
        Args:
            pos_tag (str): NLTK POS tag
            
        Returns:
            str: WordNet POS tag
        """
        # Map NLTK POS tags to WordNet POS tags
        pos = 'n'  # default to noun
        if pos_tag.startswith('J'):
            pos = 'a'  # adjective
        elif pos_tag.startswith('V'):
            pos = 'v'  # verb
        elif pos_tag.startswith('R'):
            pos = 'r'  # adverb
        
        return pos 