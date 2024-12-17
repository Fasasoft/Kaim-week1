import pandas as pd

# Load the data
raw_analyst_ratings = pd.read_csv("raw_analyst_ratings.csv")
AAPL_hISTRORICALdata = pd.read_csv("AAPL_hISTRORICALdata.csv")

# Display the first few rows of both dataframes to verify their structure
print(raw_analyst_ratings.head())
print(AAPL_hISTRORICALdata.head())

# Count the number of headlines by publisher
publisher_counts = raw_analyst_ratings['publisher'].value_counts()

# Display the top publishers by the number of headlines
print(publisher_counts.head())

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment (positive, negative, neutral)
def get_sentiment(headline):
    score = analyzer.polarity_scores(headline)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis on the headlines
raw_analyst_ratings['sentiment'] = raw_analyst_ratings['headline'].apply(get_sentiment)

# Group by publisher and get the count of sentiment types
publisher_sentiment = raw_analyst_ratings.groupby(['publisher', 'sentiment']).size().unstack(fill_value=0)

# Display sentiment breakdown for top publishers
print(publisher_sentiment.head())

# Extract the domain from the email addresses in the publisher column
raw_analyst_ratings['publisher_domain'] = raw_analyst_ratings['publisher'].str.extract(r'@([A-Za-z0-9.-]+)')

# Count the number of headlines by publisher domain
domain_counts = raw_analyst_ratings['publisher_domain'].value_counts()

# Display the top domains contributing the most articles
print(domain_counts.head())

# Ensure the 'date' columns are in datetime format
raw_analyst_ratings['date'] = pd.to_datetime(raw_analyst_ratings['date'])
AAPL_hISTRORICALdata['date'] = pd.to_datetime(AAPL_hISTRORICALdata['date'])

# Merge the two datasets on the 'date' column
merged_data = pd.merge(raw_analyst_ratings, AAPL_hISTRORICALdata[['date', 'close']], on='date', how='left')

# Display the merged data
print(merged_data[['date', 'publisher', 'headline', 'sentiment', 'close']].head())

# Save the sentiment and publisher data to a new CSV
raw_analyst_ratings.to_csv("updated_raw_analyst_ratings.csv", index=False)
 import matplotlib.pyplot as plt

# Plot the number of articles published by each publisher
plt.figure(figsize=(12, 6))
publisher_counts.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Publishers Contributing to News Feed')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
