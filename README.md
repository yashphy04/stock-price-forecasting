# Stock Predictor (Updated)

This project trains a simple XGBoost model on the provided stock CSV and predicts the next closing price.

## Setup
```bash
python -m venv venv
# activate venv
# Windows (PowerShell): venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

## Notes
- The CSV is expected to have a timestamp column (detected automatically) and `close` column as target.
- Scripts are written so you can run either `python -m scripts.train` or `python scripts/train.py` directly.

## Usage
1. Train:
```bash
python scripts/train.py
```
2. Predict:
```bash
python scripts/predict.py
```
