import numpy as np
import pandas as pd

# adapted from an older project with Johan Naresh
# https://github.com/johannaresh/CSC-project/blob/main/tickerdataframe.py

CLOSE = "Adj_Close"


def _safe_div(num, den):
    # zero denominators give NaN instead of inf so the expanding z-score in grid.py stays finite
    return num / den.where(den != 0)


def _true_range(df, close):
    prev = df[close].shift()
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev).abs()
    tr3 = (df["Low"] - prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def sma(df, n, close=CLOSE):  # Simple Moving Average
    return df[close].rolling(window=n).mean()


def ema(df, n, close=CLOSE):  # Exponential Moving Average seeded with the first value
    return df[close].ewm(alpha=2 / (n + 1), adjust=False).mean()


def rsi(df, n=14, close=CLOSE):  # Relative Strength Index
    delta = df[close].diff()
    avg_gain = delta.clip(lower=0).rolling(window=n).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(window=n).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def atr(df, n=14, close=CLOSE):  # Average True Range
    return _true_range(df, close).rolling(window=n).mean()


# OBV increases when price rises on high volume, decreases when it falls on high volume
def obv(df, close=CLOSE, volume="Volume"):
    direction = np.sign(df[close].diff()).fillna(0)
    return (direction * df[volume]).cumsum()


# shows where the current price sits within its volatility range (0 = lower band, 1 = upper band)
def bollinger_b(df, n=20, k=2, close=CLOSE):
    mid = df[close].rolling(window=n).mean()
    std = df[close].rolling(window=n).std()
    lower = mid - k * std
    return _safe_div(df[close] - lower, 2 * k * std)


def adx(df, n=14, close=CLOSE):  # Average Directional Index
    up = df["High"].diff()
    down = -df["Low"].diff()

    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr = _true_range(df, close).rolling(window=n).mean()
    plus_di = 100 * _safe_div(plus_dm.rolling(window=n).mean(), tr)
    minus_di = 100 * _safe_div(minus_dm.rolling(window=n).mean(), tr)

    dx = 100 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    return dx.rolling(window=n).mean()


def rolling_vol(df, n=20, close=CLOSE):  # Rolling Volatility (Std Dev of returns)
    return df[close].pct_change().rolling(window=n).std()


def zscore(df, n=20, close=CLOSE):  # Z-Score vs SMA(n)
    mid = df[close].rolling(window=n).mean()
    std = df[close].rolling(window=n).std()
    return _safe_div(df[close] - mid, std)


# names used by Grid and the query layer, mapped to their builders
INDICATORS = {
    "SMA5": lambda df, close: sma(df, 5, close),
    "SMA20": lambda df, close: sma(df, 20, close),
    "EMA5": lambda df, close: ema(df, 5, close),
    "EMA20": lambda df, close: ema(df, 20, close),
    "RSI14": lambda df, close: rsi(df, 14, close),
    "ATR14": lambda df, close: atr(df, 14, close),
    "OBV": lambda df, close: obv(df, close),
    "BOLL_B20": lambda df, close: bollinger_b(df, 20, 2, close),
    "ADX14": lambda df, close: adx(df, 14, close),
    "VOL20": lambda df, close: rolling_vol(df, 20, close),
    "Z20": lambda df, close: zscore(df, 20, close),
}


def indicators(df, names=None, close=CLOSE):
    # warmup rows are left as NaN rather than dropped so T matches the price series
    names = list(INDICATORS) if names is None else list(names)
    unknown = [n for n in names if n not in INDICATORS]
    if unknown:
        raise KeyError(f"unknown indicators: {unknown}, available: {list(INDICATORS)}")

    df = df.copy()
    for name in names:
        df[name] = INDICATORS[name](df, close)
    return df