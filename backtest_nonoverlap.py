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

n = len(df)
test = df.iloc[int(n * 0.8):].copy()

model = xgb.XGBClassifier()
model.load_model("model_big.json")

test["probability"] = model.predict_proba(test[FEATURE_COLS])[:, 1]

test["action"] = np.select(
    [
        test["probability"] >= BUY_THRESHOLD,
        test["probability"] <= SELL_THRESHOLD,
    ],
    [1, -1],
    default=0,
)

# One decision every 30 seconds.
sample = test.iloc[::HORIZON].copy()

sample["future_return"] = (
    sample["close"].shift(-HORIZON) / sample["close"] - 1.0
)

sample = sample.dropna(subset=["future_return"])

sample["strategy_return"] = (
    sample["action"] * sample["future_return"]
)

traded = sample[sample["action"] != 0]

print("=== NON-OVERLAPPING 30s BACKTEST ===")
print(f"test decisions: {len(sample):,}")
print(f"traded decisions: {len(traded):,}")
print(f"BUY: {(sample['action'] == 1).sum():,}")
print(f"SELL: {(sample['action'] == -1).sum():,}")
print(f"HOLD: {(sample['action'] == 0).sum():,}")

print(f"Mean trade return: {traded['strategy_return'].mean():.8f}")
print(f"Sum trade return: {traded['strategy_return'].sum():.8f}")
print(f"Win rate: {(traded['strategy_return'] > 0).mean():.6f}")

equity = (1.0 + traded["strategy_return"]).cumprod()
peak = equity.cummax()
drawdown = equity / peak - 1.0

print(f"Final equity multiple: {equity.iloc[-1]:.6f}")
print(f"Max drawdown: {drawdown.min():.6f}")
