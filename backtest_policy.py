import numpy as np
import pandas as pd
import xgboost as xgb

FEATURE_COLS = ["realized_vol", "trade_intensity", "avg_trade_size"]
HORIZON = 30

BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45

df = pd.read_parquet("features.parquet")

future_close = df["close"].shift(-HORIZON)
df["label"] = (future_close > df["close"]).astype(int)
df = df.iloc[:-HORIZON].copy()

# Same chronological split as train_model.py
n = len(df)
calib_end = int(n * 0.8)
test = df.iloc[calib_end:].copy()

model = xgb.XGBClassifier()
model.load_model("model_big.json")

test["probability"] = model.predict_proba(test[FEATURE_COLS])[:, 1]

test["action"] = np.select(
    [
        test["probability"] >= BUY_THRESHOLD,
        test["probability"] <= SELL_THRESHOLD,
    ],
    [
        1,
        -1,
    ],
    default=0,
)

test["future_return"] = (
    test["close"].shift(-HORIZON) / test["close"] - 1.0
)

test["strategy_return"] = test["action"] * test["future_return"]

test = test.dropna(subset=["future_return"])

print("=== HFT POLICY BACKTEST ===")
print(f"test rows: {len(test):,}")
print(f"BUY:  {(test['action'] == 1).sum():,}")
print(f"HOLD: {(test['action'] == 0).sum():,}")
print(f"SELL: {(test['action'] == -1).sum():,}")

print(f"Mean strategy return: {test['strategy_return'].mean():.8f}")
print(f"Sum strategy return:  {test['strategy_return'].sum():.8f}")

print(
    f"Directional accuracy: "
    f"{(test['action'] * (2 * test['label'] - 1) > 0).mean():.4f}"
)

equity = (1.0 + test["strategy_return"]).cumprod()
peak = equity.cummax()
drawdown = equity / peak - 1.0

print(f"Final equity multiple: {equity.iloc[-1]:.6f}")
print(f"Max drawdown: {drawdown.min():.6f}")
