import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
analyst_ratings = pd.read_csv("raw_analyst_ratings.csv")
historical_data = pd.read_csv("AAPL_historicaldata.csv")

# Convert the 'date' columns to datetime format
analyst_ratings['date'] = pd.to_datetime(analyst_ratings['date'])
historical_data['date'] = pd.to_datetime(historical_data['date'])

# Display the first few rows to ensure data is loaded correctly
print(analyst_ratings.head())
print(historical_data.head())

# Group the news data by date and count articles per day
daily_publications = analyst_ratings.groupby(analyst_ratings['date'].dt.date).size()

# Plot the number of publications per day
plt.figure(figsize=(12, 6))
plt.plot(daily_publications.index, daily_publications.values)
plt.title("Number of Articles Published Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Articles Published")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate the daily percentage change in stock price
historical_data['price_change'] = historical_data['close'].pct_change() * 100

# Find days with large price movements (e.g., more than 5% change)
significant_price_changes = historical_data[historical_data['price_change'].abs() > 5]  # 5% change threshold

# Display the significant price change days
print(significant_price_changes[['date', 'price_change']])

# Merge the article data with stock data based on 'date'
merged_data = pd.merge(analyst_ratings, historical_data[['date', 'close', 'price_change']], on='date', how='left')

# Group by date to get the number of articles and maximum price change for each day
publication_and_price = merged_data.groupby(merged_data['date'].dt.date).agg(
    article_count=('headline', 'size'),
    max_price_change=('price_change', 'max')
).reset_index()

# Display the data with article counts and price changes
print(publication_and_price.head())

# Extract the hour of publication from the 'date' column
analyst_ratings['hour'] = analyst_ratings['date'].dt.hour

# Count the number of articles published by hour
hourly_publications = analyst_ratings.groupby(analyst_ratings['hour']).size()

# Plot the number of articles published by hour of the day
plt.figure(figsize=(12, 6))
plt.bar(hourly_publications.index, hourly_publications.values)
plt.title("Number of Articles Published by Hour of the Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Articles Published")
plt.xticks(range(24))
plt.tight_layout()
plt.show()

# Extract the weekday from the 'date' column (0 = Monday, 6 = Sunday)
analyst_ratings['weekday'] = analyst_ratings['date'].dt.weekday

# Count the number of articles published on each weekday
weekday_publications = analyst_ratings.groupby(analyst_ratings['weekday']).size()

# Plot the number of articles published by weekday
plt.figure(figsize=(12, 6))
plt.bar(weekday_publications.index, weekday_publications.values)
plt.title("Number of Articles Published by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Number of Articles Published")
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.tight_layout()
plt.show()
