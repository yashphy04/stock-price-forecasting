import sys, pathlib
import joblib
from sklearn.metrics import mean_squared_error

# project root ko sys.path me add karna zaroori hai
project_root = pathlib.Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.preprocess import load_data
from src.features import add_features
from src.model import get_model


def main():
    print("🔄 Loading data...")
    df = load_data("data/raw/stock_data.csv")
    df = add_features(df)

    features = ['open', 'high', 'low', 'sma_5', 'sma_10',
                'volatility_10', 'hl_range', 'volume']
    features = [f for f in features if f in df.columns]
    target = 'close'

    X = df[features]
    y = df[target]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("⚡ Training model...")
    model = get_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"✅ Training complete. Test MSE: {mse:.4f}")

    joblib.dump(model, "models/xgb_model.pkl")
    print("💾 Model saved at models/xgb_model.pkl")

    last_row = df[features].iloc[-1:].values
    next_pred = model.predict(last_row)[0]
    print(f"\n📈 Next Predicted Close: {next_pred:.2f}")


if __name__ == "__main__":
    main()
