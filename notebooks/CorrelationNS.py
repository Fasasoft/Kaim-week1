import pandas as pd
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the datasets
news_df = pd.read_csv('raw_analyst_ratings.csv')
stock_df = pd.read_csv('AMZN_historical_data.csv')

# 1. **Normalize Dates and Merge Data**
# Convert dates to datetime objects
news_df['date'] = pd.to_datetime(news_df['date'])
stock_df['date'] = pd.to_datetime(stock_df['date'])

# Merge the datasets on date
merged_df = pd.merge(news_df, stock_df, on='date', how='inner')

# 2. **Sentiment Analysis on Headlines**
def get_sentiment(headline):
    analysis = TextBlob(headline)
    return analysis.sentiment.polarity  # A value between -1 (negative) and 1 (positive)

# Apply sentiment analysis to the headlines
merged_df['sentiment_score'] = merged_df['headline'].apply(get_sentiment)

# 3. **Calculate Daily Stock Returns**
# Calculate daily percentage change in the stock's closing price
stock_df['daily_return'] = stock_df['close'].pct_change() * 100  # Percentage change

# Merge the sentiment scores with daily returns
merged_df['daily_return'] = stock_df['daily_return']

# 4. **Aggregate Sentiments by Day**
# Group by date and calculate the average sentiment score for each day
daily_sentiment = merged_df.groupby('date')['sentiment_score'].mean().reset_index()

# Merge the daily sentiment with stock data
final_df = pd.merge(daily_sentiment, stock_df[['date', 'daily_return']], on='date')

# 5. **Correlation Analysis: Pearson Correlation**
# Calculate the Pearson correlation between sentiment and stock returns
correlation, _ = pearsonr(final_df['sentiment_score'], final_df['daily_return'])

print(f"Pearson correlation between sentiment and stock return: {correlation:.2f}")

# 6. **Visualizing the Correlation**
plt.scatter(final_df['sentiment_score'], final_df['daily_return'])
plt.title('Sentiment vs Stock Returns')
plt.xlabel('Sentiment Score')
plt.ylabel('Daily Stock Return (%)')
plt.show()
