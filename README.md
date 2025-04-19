# WallStreetBets Sentiment Analysis

A web application that analyzes sentiment and topics from r/wallstreetbets posts using natural language processing and machine learning techniques.

## Features

- **Sentiment Analysis**: Analyzes the sentiment of posts using VADER sentiment analysis
- **Topic Modeling**: Identifies main topics in posts using LDA (Latent Dirichlet Allocation)
- **Ticker Extraction**: Automatically identifies stock tickers mentioned in posts
- **Trend Analysis**: Visualizes trends in ticker mentions, sentiment, and topics over time
- **Interactive UI**: Modern, responsive interface with dark/light mode toggle

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A Reddit API account (for accessing r/wallstreetbets data)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/wallstreetbets-sentiment.git
   cd wallstreetbets-sentiment
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Reddit API credentials:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_user_agent
   ```

   To obtain these credentials:
   1. Go to https://www.reddit.com/prefs/apps
   2. Click "Create another app..."
   3. Select "script" as the application type
   4. Fill in the required information
   5. After creation, note the client ID (under the app name) and client secret

### Running the Application

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage Guide

### Main Dashboard

The main dashboard displays:
- Recent posts from r/wallstreetbets
- Sentiment analysis results for each post
- Identified stock tickers
- Topic modeling results

### Sentiment Analysis

- Each post is analyzed for sentiment (positive, negative, or neutral)
- Sentiment scores are displayed with color-coded badges
- The sentiment analysis uses VADER, which is specifically tuned for social media text

### Topic Modeling

- The application automatically identifies main topics in the posts
- Topics are displayed with their associated keywords
- Each post is assigned to the most relevant topic

### Ticker Extraction

- Stock tickers are automatically extracted from post content
- Tickers are displayed as badges on each post
- The system recognizes common stock symbols and filters out false positives

### Trend Analysis

- Visualizes trends in ticker mentions over time
- Shows sentiment trends for specific tickers
- Displays topic popularity trends
- Provides interactive plots and data tables

### UI Features

- **Dark/Light Mode**: Toggle between dark and light themes using the switch in the header
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Elements**: Hover over elements for additional information
- **Data Tables**: Sort and filter data in the trend analysis tables

## Troubleshooting

### Common Issues

1. **Reddit API Authentication Errors**:
   - Verify your Reddit API credentials in the `.env` file
   - Ensure your Reddit account has the necessary permissions

2. **NLTK Data Missing**:
   - The application will attempt to download required NLTK data automatically
   - If issues persist, manually download the data:
     ```python
     import nltk
     nltk.download('punkt')
     nltk.download('averaged_perceptron_tagger')
     nltk.download('wordnet')
     nltk.download('vader_lexicon')
     ```

3. **Application Not Starting**:
   - Check that all dependencies are installed correctly
   - Verify that the Flask application is running on the correct port
   - Check the console for error messages

## Project Structure

- `app.py`: Main Flask application
- `text_processor.py`: Text processing and sentiment analysis
- `topic_modeler.py`: Topic modeling using LDA
- `trend_analyzer.py`: Trend analysis and visualization
- `templates/`: HTML templates
- `static/`: CSS, JavaScript, and other static files

## Technologies Used

- **Flask**: Web framework
- **PRAW**: Reddit API wrapper
- **NLTK**: Natural language processing
- **VADER**: Sentiment analysis
- **Gensim**: Topic modeling
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive plots

## Future Enhancements

- Add more advanced trend analysis features
- Implement machine learning for stock price prediction
- Add user authentication for personalized dashboards
- Expand data sources beyond Reddit
- Implement real-time analysis with WebSockets

## License

This project is licensed under the MIT License - see the LICENSE file for details. 