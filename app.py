"""
WallStreetBets Sentiment Analysis and Topic Modeling Application

This Flask application fetches posts from the r/wallstreetbets subreddit,
performs sentiment analysis and topic modeling on the posts, and displays
the results in a web interface.

The application uses:
- Reddit API (via PRAW) to fetch posts
- NLTK's VADER sentiment analyzer for sentiment analysis
- Custom text processing for cleaning and tokenization
- Latent Dirichlet Allocation (LDA) for topic modeling
- Pandas for data manipulation and trend analysis
- Matplotlib and Seaborn for data visualization
"""

import os
from flask import Flask, render_template, request, jsonify
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from text_processor import TextProcessor
from topic_modeler import TopicModeler
from trend_analyzer import TrendAnalyzer
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
logger.info("Downloading required NLTK data...")
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
logger.info("NLTK data downloaded successfully")

# Initialize Flask application
app = Flask(__name__)

# Add custom Jinja2 filter for datetime formatting
@app.template_filter('format_datetime')
def format_datetime(timestamp):
    """Convert Unix timestamp to formatted datetime string."""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

# Initialize Reddit API client using credentials from environment variables
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

# Initialize text processor, sentiment analyzer, topic modeler, and trend analyzer
text_processor = TextProcessor()
sentiment_analyzer = SentimentIntensityAnalyzer()
topic_modeler = TopicModeler(n_topics=5)  # Extract 5 topics from the posts
trend_analyzer = TrendAnalyzer()

def get_wsb_posts():
    """
    Fetch top posts from r/wallstreetbets from the last 24 hours.
    
    This function:
    1. Retrieves the top 25 posts from the subreddit
    2. Processes each post title for text analysis
    3. Extracts stock tickers mentioned in the title
    4. Performs sentiment analysis on the title
    5. Returns a list of dictionaries containing post data and analysis results
    
    Returns:
        list: List of dictionaries containing post data and analysis results
    """
    subreddit = reddit.subreddit('wallstreetbets')
    posts = []
    
    # Get posts from the last 24 hours
    for post in subreddit.top(time_filter='day', limit=25):
        # Process the title for text analysis
        processed_title = text_processor.process_text(post.title)
        
        # Extract stock tickers from the title
        tickers = text_processor.extract_tickers(post.title)
        
        # Get sentiment scores using VADER sentiment analyzer
        sentiment_scores = sentiment_analyzer.polarity_scores(post.title)
        
        # Determine sentiment label based on compound score
        if sentiment_scores['compound'] > 0.05:
            sentiment_label = 'positive'
        elif sentiment_scores['compound'] < -0.05:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # Create a dictionary with post data and analysis results
        posts.append({
            'id': post.id,
            'title': post.title,
            'processed_text': ' '.join(processed_title),
            'tickers': tickers,
            'score': post.score,
            'num_comments': post.num_comments,
            'url': f"https://reddit.com{post.permalink}",
            'created_utc': post.created_utc,
            'sentiment': {
                'compound': sentiment_scores['compound'],
                'pos': sentiment_scores['pos'],
                'neg': sentiment_scores['neg'],
                'neu': sentiment_scores['neu'],
                'label': sentiment_label
            }
        })
    
    return posts

@app.route('/')
def index():
    """
    Render the main page with WSB posts, sentiment analysis, topic modeling, and trend analysis.
    
    This route:
    1. Fetches posts from r/wallstreetbets
    2. Performs sentiment analysis on the posts
    3. Performs topic modeling on the post titles
    4. Calculates statistics about sentiment and ticker mentions
    5. Performs trend analysis on the posts
    6. Renders the index.html template with the results
    
    Returns:
        str: Rendered HTML template with post data and analysis results
    """
    # Fetch posts from r/wallstreetbets
    posts = get_wsb_posts()
    
    # Convert to DataFrame for easier manipulation
    # Flatten the sentiment dictionary to make it easier to work with
    flattened_posts = []
    for post in posts:
        flattened_post = post.copy()
        # Extract sentiment values from the nested dictionary
        flattened_post['sentiment_compound'] = post['sentiment']['compound']
        flattened_post['sentiment_pos'] = post['sentiment']['pos']
        flattened_post['sentiment_neg'] = post['sentiment']['neg']
        flattened_post['sentiment_neu'] = post['sentiment']['neu']
        flattened_post['sentiment_label'] = post['sentiment']['label']
        flattened_posts.append(flattened_post)
    
    df = pd.DataFrame(flattened_posts)
    
    # Sort by compound sentiment score (most positive first)
    df = df.sort_values('sentiment_compound', ascending=False)
    
    # Calculate overall sentiment statistics
    sentiment_stats = {
        'total_posts': len(df),
        'positive_posts': len(df[df['sentiment_compound'] > 0.05]),
        'negative_posts': len(df[df['sentiment_compound'] < -0.05]),
        'neutral_posts': len(df[(df['sentiment_compound'] >= -0.05) & (df['sentiment_compound'] <= 0.05)]),
        'avg_sentiment': df['sentiment_compound'].mean()
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
    
    # Sort tickers by frequency (most mentioned first)
    sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Perform topic modeling on post titles
    if len(df) > 0:
        # Fit the topic modeler on the post titles
        topic_modeler.fit(df['title'].tolist())
        
        # Get document topics (most likely topic for each post)
        document_topics = topic_modeler.get_document_topics(df['title'].tolist())
        
        # Add topic information to the posts
        for i, post in enumerate(posts):
            doc, topic_idx, topic_prob = document_topics[i]
            post['topic_idx'] = topic_idx
            post['topic_prob'] = topic_prob
            post['topic'] = f"Topic {topic_idx+1}"
            post['topic_keywords'] = topic_modeler.get_topic_keywords(topic_idx)
        
        # Get all topics for display
        all_topics = []
        for i in range(topic_modeler.n_topics):
            all_topics.append({
                'idx': i,
                'keywords': topic_modeler.get_topic_keywords(i)
            })
    else:
        all_topics = []
    
    # Perform trend analysis
    trend_data = {}
    if len(posts) > 0:
        # Create DataFrame for trend analysis
        trend_analyzer.create_dataframe(posts)
        
        # Get ticker trends
        trend_data['ticker_trends'] = trend_analyzer.get_ticker_trends(top_n=5, time_period='hour')
        
        # Get topic trends
        trend_data['topic_trends'] = trend_analyzer.get_topic_trends(time_period='hour')
        
        # Get sentiment trends
        trend_data['sentiment_trends'] = trend_analyzer.get_sentiment_trends(time_period='hour')
        
        # Generate plots
        trend_data['ticker_mentions_plot'] = trend_analyzer.plot_ticker_mentions(top_n=5, time_period='hour')
        trend_data['ticker_sentiment_plot'] = trend_analyzer.plot_ticker_sentiment(top_n=5, time_period='hour')
        trend_data['topic_mentions_plot'] = trend_analyzer.plot_topic_mentions(time_period='hour')
        trend_data['sentiment_trend_plot'] = trend_analyzer.plot_sentiment_trend(time_period='hour')
    
    # Render the template with all the data
    return render_template('index.html', 
                          posts=posts, 
                          stats=sentiment_stats, 
                          tickers=sorted_tickers,
                          topics=all_topics,
                          trend_data=trend_data)

@app.route('/health')
def health_check():
    """Health check endpoint for the load balancer."""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)