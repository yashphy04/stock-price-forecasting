import joblib
import pandas as pd
from src.preprocess import load_data
from src.features import add_features

def predict_next():
    model = joblib.load('models/xgb_model.pkl')
    df = load_data('data/raw/stock_data.csv')
    df = add_features(df)
    features = ['open','high','low','sma_5','sma_10','volatility_10','hl_range','volume']
    features = [f for f in features if f in df.columns]
    last_row = df[features].iloc[-1:].values
    pred = model.predict(last_row)
    print(f"📈 Next predicted Close: {pred[0]:.2f}")
    return pred[0]

if __name__ == '__main__':
    predict_next()

