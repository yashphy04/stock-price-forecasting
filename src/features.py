import pandas as pd

def add_features(df: pd.DataFrame):
    # ensure columns exist (lowercase)
    # basic price features
    df['return'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['volatility_10'] = df['return'].rolling(10).std()
    # volume features - try common names
    if 'tottrdqty' in df.columns:
        df['volume'] = df['tottrdqty']
    elif 'volume' in df.columns:
        df['volume'] = df['volume']
    else:
        df['volume'] = 0
    # additional simple features
    df['hl_range'] = df['high'] - df['low']
    df = df.dropna().reset_index(drop=True)
    return df
