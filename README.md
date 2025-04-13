# WallStreetBets Sentiment Analysis

A web application that analyzes sentiment and topics from r/wallstreetbets posts using Reddit's API, NLTK for sentiment analysis, and Latent Dirichlet Allocation (LDA) for topic modeling.

## Overview

This application fetches posts from the r/wallstreetbets subreddit, performs sentiment analysis and topic modeling on the posts, and displays the results in a web interface. It provides insights into the sentiment of the WallStreetBets community and the topics they're discussing.

## Features

### Data Collection
- Fetches top 25 posts from r/wallstreetbets from the last 24 hours
- Extracts post titles, scores, comment counts, and URLs

### Text Processing
- Cleans text by removing URLs, emojis, special characters, and numbers
- Extracts stock tickers from post titles
- Tokenizes text into words
- Removes stop words
- Lemmatizes words to their base form

### Sentiment Analysis
- Performs sentiment analysis on post titles using NLTK's VADER sentiment analyzer
- Calculates compound, positive, negative, and neutral sentiment scores
- Categorizes posts as positive, negative, or neutral based on sentiment scores
- Provides sentiment statistics (total posts, positive posts, negative posts, neutral posts)

### Topic Modeling
- Uses Latent Dirichlet Allocation (LDA) to identify topics in post titles
- Extracts keywords for each topic with their weights
- Assigns each post to its most likely topic
- Displays topic modeling results with keywords and weights

### Ticker Analysis
- Extracts stock tickers mentioned in post titles
- Counts the frequency of each ticker
- Displays the most mentioned tickers

### User Interface
- Modern, responsive UI using Bootstrap
- Displays posts with their scores, comment counts, and sentiment scores
- Color-coded sentiment indicators (green for positive, red for negative, gray for neutral)
- Interactive cards with hover effects
- Organized sections for sentiment statistics, topic modeling results, and ticker statistics

## Setup

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your Reddit API credentials:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_app_name_v1.0
   ```

   To get these credentials:
   1. Go to https://www.reddit.com/prefs/apps
   2. Click "create another app..."
   3. Select "script"
   4. Fill in the required information
   5. Copy the client ID and client secret to your .env file

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py` - Main Flask application
- `text_processor.py` - Text processing utilities
- `topic_modeler.py` - Topic modeling using LDA
- `templates/index.html` - HTML template for the web interface
- `requirements.txt` - Python dependencies

## Technologies Used

- **Flask** - Web framework
- **PRAW** - Reddit API wrapper
- **NLTK** - Natural Language Toolkit for sentiment analysis
- **scikit-learn** - Machine learning library for topic modeling
- **Bootstrap** - Frontend framework
- **Python-dotenv** - Environment variable management
- **pandas** - Data manipulation and analysis

## Future Enhancements

- Add historical sentiment analysis
- Implement user authentication
- Add more detailed post analysis
- Include sentiment analysis for comments
- Add visualization of sentiment trends over time
- Implement real-time updates using WebSockets 