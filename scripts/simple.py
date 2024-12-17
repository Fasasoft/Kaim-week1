import yfinance as yf

# Fetch stock data for Apple (AAPL)
data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')

# Print the closing prices
print(data['Close'])
