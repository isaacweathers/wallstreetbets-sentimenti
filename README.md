# WallStreetBets Sentiment Analysis

A web application that analyzes sentiment from r/wallstreetbets posts using Reddit's API and NLTK for sentiment analysis.

## Features

- Fetches top 25 posts from r/wallstreetbets from the last 24 hours
- Performs sentiment analysis on post titles
- Displays posts with their scores, comment counts, and sentiment scores
- Modern, responsive UI using Bootstrap

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

## Technologies Used

- Flask - Web framework
- PRAW - Reddit API wrapper
- NLTK - Natural Language Toolkit for sentiment analysis
- Bootstrap - Frontend framework
- Python-dotenv - Environment variable management 