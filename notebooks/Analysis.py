import pandas as pd
import talib as ta

# Load the TSLA historical data
tsla_data = pd.read_csv("/home/user/data/TSLA_historical_data.csv")


# Convert 'date' column to datetime
tsla_data['date'] = pd.to_datetime(tsla_data['date'])

# Display the first few rows to ensure the data is loaded correctly
print(tsla_data.head())

# Calculate the 50-day Simple Moving Average (SMA)
tsla_data['SMA_50'] = ta.SMA(tsla_data['close'], timeperiod=50)

# Calculate the 200-day Simple Moving Average (SMA)
tsla_data['SMA_200'] = ta.SMA(tsla_data['close'], timeperiod=200)

# Display the updated dataframe with SMA
print(tsla_data[['date', 'close', 'SMA_50', 'SMA_200']].tail())


# Calculate the 14-day Relative Strength Index (RSI)
tsla_data['RSI_14'] = ta.RSI(tsla_data['close'], timeperiod=14)

# Display the updated dataframe with RSI
print(tsla_data[['date', 'close', 'RSI_14']].tail())


# Calculate MACD (12, 26, 9) – commonly used parameters for MACD
macd, macdsignal, macdhist = ta.MACD(tsla_data['close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Add MACD and signal line to the dataframe
tsla_data['MACD'] = macd
tsla_data['MACD_signal'] = macdsignal

# Display the updated dataframe with MACD
print(tsla_data[['date', 'close', 'MACD', 'MACD_signal']].tail())


# Calculate Bollinger Bands (20-period)
upperband, middleband, lowerband = ta.BBANDS(tsla_data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# Add Bollinger Bands to the dataframe
tsla_data['Bollinger_upper'] = upperband
tsla_data['Bollinger_middle'] = middleband
tsla_data['Bollinger_lower'] = lowerband


print(tsla_data[['date', 'close', 'Bollinger_upper', 'Bollinger_middle', 'Bollinger_lower']].tail())

# Example: On-Balance Volume (OBV)
tsla_data['OBV'] = ta.OBV(tsla_data['close'], tsla_data['volume'])

# Display the updated dataframe with OBV
print(tsla_data[['date', 'close', 'OBV']].tail())

import pynance

# Initialize the strategy with historical data
strategy = pynance.Strategy(tsla_data)

# Set up an example strategy (e.g., buy if the MACD crosses above the signal line)
strategy.add_signal('macd_cross', (tsla_data['MACD'] > tsla_data['MACD_signal']))

# Perform backtesting
results = strategy.run()

# Display the backtest results
print(results)
