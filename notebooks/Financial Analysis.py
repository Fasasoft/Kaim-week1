import pandas as pd


news_data = pd.read_csv("raw_analyst_ratings.csv")
historical_data = pd.read_csv("AAPL_historicaldata.csv")


print(news_data.head())
print(historical_data.head())
Sentiment Analysis
from textblob import TextBlob

# Function to get sentiment (positive, negative, neutral)
def get_sentiment(headline):
    analysis = TextBlob(headline)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"


news_data['sentiment'] = news_data['headline'].apply(get_sentiment)


print(news_data[['headline', 'sentiment']].head())


import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Preprocess headlines
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

processed_headlines = news_data['headline'].apply(preprocess_text)


dictionary = corpora.Dictionary(processed_headlines)
corpus = [dictionary.doc2bow(text) for text in processed_headlines]

# Apply LDA topic modeling
lda_model = gensim.models.LdaMulticore(corpus, num_topics=3, id2word=dictionary, passes=10)

# Display topics
topics = lda_model.print_topics(num_topics=3, num_words=5)
for idx, topic in topics:
    print(f"Topic {idx}: {topic}")



# Merge sentiment with historical stock data on the 'date' column
news_data['date'] = pd.to_datetime(news_data['date'])
historical_data['date'] = pd.to_datetime(historical_data['date'])

# Merge the two dataframes on date
merged_data = pd.merge(news_data, historical_data, on='date', how='left')

# Display the merged data
print(merged_data[['date', 'headline', 'sentiment', 'close']].head())

# Sentiment vs Stock Price
import matplotlib.pyplot as plt

# Plot sentiment vs. stock price
# You can map sentiment to numerical values (e.g., Positive = 1, Negative = -1, Neutral = 0)
sentiment_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
merged_data['sentiment_value'] = merged_data['sentiment'].map(sentiment_map)

# Plot sentiment vs. closing stock prices
plt.figure(figsize=(10,6))
plt.scatter(merged_data['sentiment_value'], merged_data['close'], alpha=0.5)
plt.title('Sentiment vs. Stock Price')
plt.xlabel('Sentiment (Positive = 1, Negative = -1, Neutral = 0)')
plt.ylabel('Stock Price (Close)')
plt.show()


