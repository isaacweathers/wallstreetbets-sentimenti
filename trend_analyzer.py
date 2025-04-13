"""
Trend Analyzer Module

This module provides functionality for analyzing trends in WallStreetBets posts,
including aggregation by time, ticker, and topic, as well as visualization of trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64

class TrendAnalyzer:
    """
    A class for analyzing trends in WallStreetBets posts.
    
    This class provides methods for aggregating data by time, ticker, and topic,
    calculating various metrics, and generating visualizations.
    """
    
    def __init__(self):
        """Initialize the TrendAnalyzer."""
        self.df = None
        self.time_grouped = None
        self.ticker_grouped = None
        self.topic_grouped = None
    
    def create_dataframe(self, posts):
        """
        Create a pandas DataFrame from the posts data.
        
        Args:
            posts (list): List of post dictionaries containing post data
            
        Returns:
            pandas.DataFrame: DataFrame containing the post data
        """
        # Extract relevant data from posts
        data = []
        for post in posts:
            # Convert timestamp to datetime
            created_utc = datetime.fromtimestamp(post['created_utc'])
            
            # Extract tickers and topics
            tickers = post.get('tickers', [])
            topic = post.get('topic', 'Unknown')
            topic_keywords = post.get('topic_keywords', [])
            
            # Create a row for each ticker mentioned in the post
            if tickers:
                for ticker in tickers:
                    data.append({
                        'id': post['id'],
                        'title': post['title'],
                        'processed_text': post['processed_text'],
                        'score': post['score'],
                        'num_comments': post['num_comments'],
                        'created_utc': created_utc,
                        'date': created_utc.date(),
                        'hour': created_utc.hour,
                        'ticker': ticker,
                        'topic': topic,
                        'topic_keywords': topic_keywords,
                        'sentiment_compound': post['sentiment']['compound'],
                        'sentiment_pos': post['sentiment']['pos'],
                        'sentiment_neg': post['sentiment']['neg'],
                        'sentiment_neu': post['sentiment']['neu'],
                        'sentiment_label': post['sentiment']['label']
                    })
            else:
                # If no tickers, still include the post with a placeholder ticker
                data.append({
                    'id': post['id'],
                    'title': post['title'],
                    'processed_text': post['processed_text'],
                    'score': post['score'],
                    'num_comments': post['num_comments'],
                    'created_utc': created_utc,
                    'date': created_utc.date(),
                    'hour': created_utc.hour,
                    'ticker': 'NO_TICKER',
                    'topic': topic,
                    'topic_keywords': topic_keywords,
                    'sentiment_compound': post['sentiment']['compound'],
                    'sentiment_pos': post['sentiment']['pos'],
                    'sentiment_neg': post['sentiment']['neg'],
                    'sentiment_neu': post['sentiment']['neu'],
                    'sentiment_label': post['sentiment']['label']
                })
        
        # Create DataFrame
        self.df = pd.DataFrame(data)
        
        # Group data by time, ticker, and topic
        self._group_data()
        
        return self.df
    
    def _group_data(self):
        """Group data by time, ticker, and topic for analysis."""
        if self.df is None or self.df.empty:
            return
        
        # Group by date
        self.time_grouped = self.df.groupby('date')
        
        # Group by ticker
        self.ticker_grouped = self.df.groupby('ticker')
        
        # Group by topic
        self.topic_grouped = self.df.groupby('topic')
    
    def get_ticker_trends(self, top_n=5, time_period='day'):
        """
        Get trends for the top N tickers over time.
        
        Args:
            top_n (int): Number of top tickers to analyze
            time_period (str): Time period for grouping ('hour', 'day')
            
        Returns:
            dict: Dictionary containing trend data for top tickers
        """
        if self.df is None or self.df.empty:
            return {}
        
        # Get top N tickers by mention count
        top_tickers = self.df['ticker'].value_counts().head(top_n).index.tolist()
        
        # Filter data for top tickers
        top_ticker_data = self.df[self.df['ticker'].isin(top_tickers)]
        
        # Group by time period and ticker
        if time_period == 'hour':
            grouped = top_ticker_data.groupby(['hour', 'ticker'])
        else:  # day
            grouped = top_ticker_data.groupby(['date', 'ticker'])
        
        # Calculate metrics
        mention_counts = grouped.size().unstack(fill_value=0)
        avg_sentiment = grouped['sentiment_compound'].mean().unstack(fill_value=0)
        sum_scores = grouped['score'].sum().unstack(fill_value=0)
        
        # Prepare result
        result = {
            'top_tickers': top_tickers,
            'mention_counts': mention_counts.to_dict(orient='index'),
            'avg_sentiment': avg_sentiment.to_dict(orient='index'),
            'sum_scores': sum_scores.to_dict(orient='index'),
            'time_period': time_period
        }
        
        return result
    
    def get_topic_trends(self, time_period='day'):
        """
        Get trends for topics over time.
        
        Args:
            time_period (str): Time period for grouping ('hour', 'day')
            
        Returns:
            dict: Dictionary containing trend data for topics
        """
        if self.df is None or self.df.empty:
            return {}
        
        # Group by time period and topic
        if time_period == 'hour':
            grouped = self.df.groupby(['hour', 'topic'])
        else:  # day
            grouped = self.df.groupby(['date', 'topic'])
        
        # Calculate metrics
        mention_counts = grouped.size().unstack(fill_value=0)
        avg_sentiment = grouped['sentiment_compound'].mean().unstack(fill_value=0)
        sum_scores = grouped['score'].sum().unstack(fill_value=0)
        
        # Prepare result
        result = {
            'topics': self.df['topic'].unique().tolist(),
            'mention_counts': mention_counts.to_dict(orient='index'),
            'avg_sentiment': avg_sentiment.to_dict(orient='index'),
            'sum_scores': sum_scores.to_dict(orient='index'),
            'time_period': time_period
        }
        
        return result
    
    def get_sentiment_trends(self, time_period='day'):
        """
        Get sentiment trends over time.
        
        Args:
            time_period (str): Time period for grouping ('hour', 'day')
            
        Returns:
            dict: Dictionary containing sentiment trend data
        """
        if self.df is None or self.df.empty:
            return {}
        
        # Group by time period
        if time_period == 'hour':
            grouped = self.df.groupby('hour')
        else:  # day
            grouped = self.df.groupby('date')
        
        # Calculate metrics
        avg_compound = grouped['sentiment_compound'].mean()
        avg_positive = grouped['sentiment_pos'].mean()
        avg_negative = grouped['sentiment_neg'].mean()
        avg_neutral = grouped['sentiment_neu'].mean()
        
        # Count sentiment labels
        sentiment_counts = grouped['sentiment_label'].value_counts().unstack(fill_value=0)
        
        # Prepare result
        result = {
            'time_period': time_period,
            'avg_compound': avg_compound.to_dict(),
            'avg_positive': avg_positive.to_dict(),
            'avg_negative': avg_negative.to_dict(),
            'avg_neutral': avg_neutral.to_dict(),
            'sentiment_counts': sentiment_counts.to_dict(orient='index')
        }
        
        return result
    
    def plot_ticker_mentions(self, top_n=5, time_period='day'):
        """
        Plot mention counts for top N tickers over time.
        
        Args:
            top_n (int): Number of top tickers to plot
            time_period (str): Time period for grouping ('hour', 'day')
            
        Returns:
            str: Base64 encoded PNG image of the plot
        """
        if self.df is None or self.df.empty:
            return ""
        
        # Get top N tickers
        top_tickers = self.df['ticker'].value_counts().head(top_n).index.tolist()
        
        # Filter data for top tickers
        top_ticker_data = self.df[self.df['ticker'].isin(top_tickers)]
        
        # Group by time period and ticker
        if time_period == 'hour':
            grouped = top_ticker_data.groupby(['hour', 'ticker']).size().unstack(fill_value=0)
            x_label = 'Hour of Day'
        else:  # day
            grouped = top_ticker_data.groupby(['date', 'ticker']).size().unstack(fill_value=0)
            x_label = 'Date'
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        for ticker in top_tickers:
            plt.plot(grouped.index, grouped[ticker], marker='o', label=ticker)
        
        plt.title(f'Top {top_n} Ticker Mentions Over Time')
        plt.xlabel(x_label)
        plt.ylabel('Number of Mentions')
        plt.legend()
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    
    def plot_ticker_sentiment(self, top_n=5, time_period='day'):
        """
        Plot average sentiment for top N tickers over time.
        
        Args:
            top_n (int): Number of top tickers to plot
            time_period (str): Time period for grouping ('hour', 'day')
            
        Returns:
            str: Base64 encoded PNG image of the plot
        """
        if self.df is None or self.df.empty:
            return ""
        
        # Get top N tickers
        top_tickers = self.df['ticker'].value_counts().head(top_n).index.tolist()
        
        # Filter data for top tickers
        top_ticker_data = self.df[self.df['ticker'].isin(top_tickers)]
        
        # Group by time period and ticker
        if time_period == 'hour':
            grouped = top_ticker_data.groupby(['hour', 'ticker'])['sentiment_compound'].mean().unstack(fill_value=0)
            x_label = 'Hour of Day'
        else:  # day
            grouped = top_ticker_data.groupby(['date', 'ticker'])['sentiment_compound'].mean().unstack(fill_value=0)
            x_label = 'Date'
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        for ticker in top_tickers:
            plt.plot(grouped.index, grouped[ticker], marker='o', label=ticker)
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title(f'Average Sentiment for Top {top_n} Tickers Over Time')
        plt.xlabel(x_label)
        plt.ylabel('Average Sentiment Score')
        plt.legend()
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    
    def plot_topic_mentions(self, time_period='day'):
        """
        Plot mention counts for topics over time.
        
        Args:
            time_period (str): Time period for grouping ('hour', 'day')
            
        Returns:
            str: Base64 encoded PNG image of the plot
        """
        if self.df is None or self.df.empty:
            return ""
        
        # Group by time period and topic
        if time_period == 'hour':
            grouped = self.df.groupby(['hour', 'topic']).size().unstack(fill_value=0)
            x_label = 'Hour of Day'
        else:  # day
            grouped = self.df.groupby(['date', 'topic']).size().unstack(fill_value=0)
            x_label = 'Date'
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        for topic in grouped.columns:
            plt.plot(grouped.index, grouped[topic], marker='o', label=topic)
        
        plt.title('Topic Mentions Over Time')
        plt.xlabel(x_label)
        plt.ylabel('Number of Mentions')
        plt.legend()
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    
    def plot_sentiment_trend(self, time_period='day'):
        """
        Plot sentiment trends over time.
        
        Args:
            time_period (str): Time period for grouping ('hour', 'day')
            
        Returns:
            str: Base64 encoded PNG image of the plot
        """
        if self.df is None or self.df.empty:
            return ""
        
        # Group by time period
        if time_period == 'hour':
            grouped = self.df.groupby('hour')
            x_label = 'Hour of Day'
        else:  # day
            grouped = self.df.groupby('date')
            x_label = 'Date'
        
        # Calculate metrics
        avg_compound = grouped['sentiment_compound'].mean()
        avg_positive = grouped['sentiment_pos'].mean()
        avg_negative = grouped['sentiment_neg'].mean()
        avg_neutral = grouped['sentiment_neu'].mean()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        plt.plot(avg_compound.index, avg_compound.values, marker='o', label='Compound', color='blue')
        plt.plot(avg_positive.index, avg_positive.values, marker='o', label='Positive', color='green')
        plt.plot(avg_negative.index, avg_negative.values, marker='o', label='Negative', color='red')
        plt.plot(avg_neutral.index, avg_neutral.values, marker='o', label='Neutral', color='gray')
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Sentiment Trends Over Time')
        plt.xlabel(x_label)
        plt.ylabel('Average Sentiment Score')
        plt.legend()
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url 