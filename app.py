import os
from flask import Flask, render_template
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from text_processor import TextProcessor
from topic_modeler import TopicModeler

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

# Initialize text processor, sentiment analyzer, and topic modeler
text_processor = TextProcessor()
sentiment_analyzer = SentimentIntensityAnalyzer()
topic_modeler = TopicModeler(n_topics=5)

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
    """Render the main page with WSB posts, sentiment analysis, and topic modeling."""
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
    
    # Perform topic modeling on post titles
    if len(df) > 0:
        # Fit the topic modeler
        topic_modeler.fit(df['title'].tolist())
        
        # Get document topics
        document_topics = topic_modeler.get_document_topics(df['title'].tolist())
        
        # Add topic information to the posts
        for i, post in enumerate(df.to_dict('records')):
            doc, topic_idx, topic_prob = document_topics[i]
            post['topic_idx'] = topic_idx
            post['topic_prob'] = topic_prob
            post['topic_keywords'] = topic_modeler.get_topic_keywords(topic_idx)
        
        # Get all topics
        all_topics = []
        for i in range(topic_modeler.n_topics):
            all_topics.append({
                'idx': i,
                'keywords': topic_modeler.get_topic_keywords(i)
            })
    else:
        all_topics = []
    
    return render_template('index.html', 
                          posts=df.to_dict('records'), 
                          stats=sentiment_stats, 
                          tickers=sorted_tickers,
                          topics=all_topics)

if __name__ == '__main__':
    app.run(debug=True) 