import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

from src.preprocess import load_data
from src.features import add_features
from src.model import get_model

def train():
    os.makedirs('models', exist_ok=True)
    df = load_data('data/raw/stock_data.csv')
    df = add_features(df)

    features = ['open','high','low','sma_5','sma_10','volatility_10','hl_range','volume']
    # ensure features exist
    features = [f for f in features if f in df.columns]
    target = 'close'

    X = df[features]
    y = df[target]

    # time-series split: keep order, use last 20% as test
    split_idx = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = get_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse:.4f}")

    joblib.dump(model, "models/xgb_model.pkl")
    print("✅ Model saved at models/xgb_model.pkl")

if __name__ == '__main__':
    train()

