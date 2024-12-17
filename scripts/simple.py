import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\Dell\Desktop\Kaim\Kaim-week1\EDA\raw_analyst_ratings.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
# Calculate headline lengths
df['headline_length'] = df['headline'].apply(lambda x: len(str(x)))
headline_stats = df['headline_length'].describe()
print("\nBasic Statistics for Headline Lengths:")
print(headline_stats)
# Plot headline lengths
df['headline_length'].plot(kind='hist', bins=30, title='Distribution of Headline Lengths')
plt.xlabel('Headline Length (characters)')
plt.show()

# --- 2. Articles per Publisher ---
# Count the number of articles per publisher
publisher_counts = df['publisher'].value_counts()
print("\nTop Publishers by Article Count:")
print(publisher_counts)

# Plot top publishers
plt.figure(figsize=(10, 6))
sns.barplot(x=publisher_counts.index[:10], y=publisher_counts.values[:10], palette='viridis')
plt.title('Top 10 Most Active Publishers')
plt.xticks(rotation=45)
plt.ylabel('Number of Articles')
plt.xlabel('Publisher')
plt.show()

# --- 3. Publication Date Trends ---
# Extract day of the week
df['day_of_week'] = df['date'].dt.day_name()

# Count articles by day of the week
day_counts = df['day_of_week'].value_counts()
print("\nArticles Published by Day of the Week:")
print(day_counts)

# Plot publication frequency by day of the week
sns.barplot(x=day_counts.index, y=day_counts.values, palette='coolwarm')
plt.title('Articles Published by Day of the Week')
plt.ylabel('Number of Articles')
plt.xlabel('Day of the Week')
plt.show()

# --- 4. Stock Column Analysis ---
# Check unique stock values
unique_stocks = df['stock'].unique()
print("\nUnique Stock Symbols:")
print(unique_stocks)

# Count articles per stock symbol
stock_counts = df['stock'].value_counts()
print("\nArticles by Stock Symbol:")
print(stock_counts)

# Plot articles by stock symbol
plt.figure(figsize=(10, 6))
sns.barplot(x=stock_counts.index[:10], y=stock_counts.values[:10], palette='plasma')
plt.title('Top 10 Stock Symbols by Article Count')
plt.xticks(rotation=45)
plt.ylabel('Number of Articles')
plt.xlabel('Stock Symbol')
plt.show()

# --- 5. Time Trends: Articles per Month ---
df['month'] = df['date'].dt.to_period('M')
monthly_counts = df['month'].value_counts().sort_index()

print("\nArticles Published Over Time (Monthly):")
print(monthly_counts)

# Plot articles over time
monthly_counts.plot(kind='line', marker='o', title='Article Publication Trend Over Time')
plt.ylabel('Number of Articles')
plt.xlabel('Publication Month')
plt.show()


