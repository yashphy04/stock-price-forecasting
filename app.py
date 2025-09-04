import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os   # 👈 add kiya

from src.preprocess import load_data
from src.features import add_features
from src.model import get_model
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="📈 Stock Predictor", layout="wide")

st.title("📊 Stock Price Prediction App")

# ✅ Load & preprocess data (absolute path use karke)
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "data", "raw", "stock_data.csv")

df = load_data(csv_path)
df = add_features(df)

features = ['open', 'high', 'low', 'sma_5', 'sma_10',
            'volatility_10', 'hl_range', 'volume']
features = [f for f in features if f in df.columns]
target = 'close'

# Sidebar options
st.sidebar.header("Options")
action = st.sidebar.radio("Choose Action:", ["Train Model", "Predict Next Close", "View Data"])

if action == "Train Model":
    st.subheader("⚡ Training Model...")
    split_idx = int(len(df) * 0.8)
    X_train, X_test = df[features].iloc[:split_idx], df[features].iloc[split_idx:]
    y_train, y_test = df[target].iloc[:split_idx], df[target].iloc[split_idx:]

    model = get_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    joblib.dump(model, os.path.join(BASE_DIR, "models", "xgb_model.pkl"))  # 👈 safe model path

    st.success(f"✅ Model trained successfully. Test MSE: {mse:.4f}")
    st.write("Model saved at: `models/xgb_model.pkl`")

elif action == "Predict Next Close":
    st.subheader("📈 Next Close Prediction")

    try:
        model = joblib.load(os.path.join(BASE_DIR, "models", "xgb_model.pkl"))  # 👈 safe load path
        last_row = df[features].iloc[-1:].values
        prediction = model.predict(last_row)[0]
        st.metric(label="Predicted Next Close", value=f"{prediction:.2f}")
    except:
        st.error("⚠️ Please train the model first!")

elif action == "View Data":
    st.subheader("📄 Stock Data Preview")
    st.dataframe(df.tail(20))

    st.subheader("📊 Closing Price Chart")
    fig, ax = plt.subplots()
    ax.plot(df['timestamp'], df['close'], label="Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    st.pyplot(fig)
