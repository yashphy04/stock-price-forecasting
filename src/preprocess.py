import pandas as pd

def load_data(path="data/raw/stock_data.csv"):
    df = pd.read_csv(path)
    # detect date-like column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    if date_col:
        df['timestamp'] = pd.to_datetime(df[date_col])
    else:
        # fallback: if 'timestamp' already exists use it, else create index-based timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            df['timestamp'] = pd.RangeIndex(start=0, stop=len(df), step=1)
            print("⚠ Warning: No date column found; using integer index as timestamp.")
    df = df.sort_values('timestamp').reset_index(drop=True)
    # lowercase columns for consistency
    df.columns = [c.lower() for c in df.columns]
    return df
