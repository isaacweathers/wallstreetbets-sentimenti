import os
from flask import Flask, render_template
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from text_processor import TextProcessor

# Download required NLTK data
nltk.download('vader_lexicon')

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

# Initialize text processor and sentiment analyzer
text_processor = TextProcessor()
sentiment_analyzer = SentimentIntensityAnalyzer()

def get_wsb_posts():
    """Fetch top posts from r/wallstreetbets from the last 24 hours."""
    subreddit = reddit.subreddit('wallstreetbets')
    posts = []
    
    # Get posts from the last 24 hours
    for post in subreddit.top(time_filter='day', limit=25):
        # Process the title
        processed_title = text_processor.process_text(post.title)
        
        # Extract tickers from the title
        tickers = text_processor.extract_tickers(post.title)
        
        # Get sentiment scores
        sentiment_scores = sentiment_analyzer.polarity_scores(post.title)
        
        posts.append({
            'title': post.title,
            'processed_title': ' '.join(processed_title),
            'tickers': tickers,
            'score': post.score,
            'num_comments': post.num_comments,
            'url': f"https://reddit.com{post.permalink}",
            'created_utc': datetime.fromtimestamp(post.created_utc),
            'compound_score': sentiment_scores['compound'],
            'positive_score': sentiment_scores['pos'],
            'negative_score': sentiment_scores['neg'],
            'neutral_score': sentiment_scores['neu']
        })
    
    return posts

@app.route('/')
def index():
    """Render the main page with WSB posts and sentiment analysis."""
    posts = get_wsb_posts()
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(posts)
    
    # Sort by compound sentiment score
    df = df.sort_values('compound_score', ascending=False)
    
    # Calculate overall sentiment statistics
    sentiment_stats = {
        'total_posts': len(df),
        'positive_posts': len(df[df['compound_score'] > 0.05]),
        'negative_posts': len(df[df['compound_score'] < -0.05]),
        'neutral_posts': len(df[(df['compound_score'] >= -0.05) & (df['compound_score'] <= 0.05)]),
        'avg_sentiment': df['compound_score'].mean()
    }
    
    # Get all unique tickers mentioned
    all_tickers = []
    for tickers in df['tickers']:
        all_tickers.extend(tickers)
    
    # Count ticker frequency
    ticker_counts = {}
    for ticker in all_tickers:
        if ticker in ticker_counts:
            ticker_counts[ticker] += 1
        else:
            ticker_counts[ticker] = 1
    
    # Sort tickers by frequency
    sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
    
    return render_template('index.html', posts=df.to_dict('records'), stats=sentiment_stats, tickers=sorted_tickers)

if __name__ == '__main__':
    app.run(debug=True) 