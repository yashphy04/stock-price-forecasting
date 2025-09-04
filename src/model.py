from xgboost import XGBRegressor

def get_model():
    return XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbosity=0
    )
