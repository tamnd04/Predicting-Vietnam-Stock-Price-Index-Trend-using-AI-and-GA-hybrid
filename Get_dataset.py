import pandas as pd
import numpy as np
from vnstock3 import Vnstock

stock = Vnstock().stock(symbol='VN30F1M', source='VCI')
VN30 = stock.listing.symbols_by_group('VN30')
df = stock.quote.history(symbol='ACB', start='2019-01-01', end='2024-11-10', interval='1D').rename_axis('ACB', inplace=False)
df['trend'] = (df['close'] > df['close'].shift(1)).astype(int)
df.loc[0, 'trend'] = None
df = df.dropna(how='any')
df.to_csv('ACB.csv', encoding="utf-8-sig", index=False)


def simple_moving_average(data, window):
    return data['close'].rolling(window=window).mean()


def weighted_moving_average(data, window):
    weights = list(range(1, window + 1))
    return data['close'].rolling(window=window).apply(lambda x: np.dot(x, weights) / sum(weights), raw=True)


def momentum(data, window):
    return data['close'].diff(window)


def stochastic_k(data, window):
    lowest_low = data['low'].rolling(window=window).min()
    highest_high = data['high'].rolling(window=window).max()
    return 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))


def stochastic_d(data, window):
    k = stochastic_k(data, window)
    return k.rolling(window=3).mean()  # Stochastic D is usually a 3-day moving average of %K


def rsi(data, window):
    delta = data['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=window).mean()
    loss = down.rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(data, signal_window, short_window=12, long_window=26):
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line - signal_line


def williams_r(data, window):
    highest_high = data['high'].rolling(window=window).max()
    lowest_low = data['low'].rolling(window=window).min()
    return -100 * ((highest_high - data['close']) / (highest_high - lowest_low))


def cci(data, window):
    tp = (data['high'] + data['low'] + data['close']) / 3  # Typical Price
    sma = tp.rolling(window=window).mean()
    mean_deviation = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mean_deviation)


def rate_of_change(data, window):
    return data['close'].pct_change(periods=window) * 100


def adx(data, window):
    high_diff = data['high'].diff()
    low_diff = -data['low'].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    tr = pd.DataFrame({'h-l': data['high'] - data['low'], 'h-c': (data['high'] - data['close'].shift()).abs(),
                       'l-c': (data['low'] - data['close'].shift()).abs()}).max(axis=1)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).sum() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(window=window).mean()


time_spans = [3, 5, 10, 15]
data = pd.read_csv("ACB.csv")
indicators = pd.DataFrame(index=data.index)
print(indicators)
data.dropna(inplace=True)
for window in time_spans:
    indicators[f'SMA_{window}'] = simple_moving_average(data, window)
    indicators[f'WMA_{window}'] = weighted_moving_average(data, window)
    indicators[f'Momentum_{window}'] = momentum(data, window)
    indicators[f'StochK_{window}'] = stochastic_k(data, window)
    indicators[f'StochD_{window}'] = stochastic_d(data, window)
    indicators[f'RSI_{window}'] = rsi(data, window)
    indicators[f'MACD_{window}'] = macd(data, window)
    indicators[f'WilliamsR_{window}'] = williams_r(data, window)
    indicators[f'CCI_{window}']  = cci(data, window)
    indicators[f'ROC_{window}'] = rate_of_change(data, window)
    indicators[f'ADX_{window}'] = adx(data, window)

indicators['Trend'] = data['trend']
indicators['Trend'] = indicators['Trend'].shift(1)
temp = pd.to_datetime(df['time'])
indicators['Year'] = pd.Series(temp.dt.year, index=data.index)
print(indicators)
df = pd.DataFrame(indicators)
df = df.dropna(how='any')
print(df.isna().sum()) # Check for any NaN values
print(df)
df.to_csv('dataset.csv', encoding="utf-8-sig", index=False)