"""
process_trades.py

Unzips Binance futures/um/monthly/trades/BTCUSDT-trades-YYYY-MM.zip files,
parses them, resamples into fixed time buckets, computes rolling features
(realized volatility, trade intensity, average trade size), and labels
volatility regimes (calm / normal / volatile) via quantile thresholds.

Output: a single parquet file with one row per time bucket, ready to feed
into the VEC (Validity Envelope Certificate) generator.

Usage:
    python process_trades.py --zip_dir /path/to/zips --out features.parquet

Expected input files (place all zips in one folder, e.g. ./zips/):
    BTCUSDT-trades-2026-06.zip
    BTCUSDT-trades-2026-07.zip
    BTCUSDT-trades-2026-08.zip
    BTCUSDT-trades-2026-09.zip

Binance futures trades.zip CSV schema (no header row):
    id, price, qty, quote_qty, time (ms epoch), is_buyer_maker
"""

import argparse
import glob
import os
import zipfile
import io

import numpy as np
import pandas as pd

TRADE_COLUMNS = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"]

BUCKET = "1s"
ROLL_WINDOW = 300


def load_trades_from_zip(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        frames = []
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            with zf.open(name) as f:
                raw = f.read()
            first_line = raw.split(b"\n", 1)[0].decode("utf-8", errors="ignore")
            has_header = "price" in first_line.lower()

            df = pd.read_csv(
                io.BytesIO(raw),
                header=0 if has_header else None,
                names=None if has_header else TRADE_COLUMNS,
            )
            if has_header:
                df.columns = [c.lower() for c in df.columns]
                rename_map = {}
                for c in df.columns:
                    if "price" in c:
                        rename_map[c] = "price"
                    elif "quote" in c and "qty" in c:
                        rename_map[c] = "quote_qty"
                    elif c == "qty" or ("qty" in c and "quote" not in c):
                        rename_map[c] = "qty"
                    elif "time" in c:
                        rename_map[c] = "time"
                    elif "maker" in c:
                        rename_map[c] = "is_buyer_maker"
                df = df.rename(columns=rename_map)

            frames.append(df[["price", "qty", "quote_qty", "time", "is_buyer_maker"]])

    out = pd.concat(frames, ignore_index=True)
    return out


def resample_trades(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["time"], unit="ms", utc=True)
    trades = trades.sort_values("timestamp").set_index("timestamp")
    trades["price"] = trades["price"].astype(float)
    trades["qty"] = trades["qty"].astype(float)
    trades["quote_qty"] = trades["quote_qty"].astype(float)

    agg = trades["price"].resample(BUCKET).ohlc()
    agg["volume"] = trades["qty"].resample(BUCKET).sum()
    agg["notional"] = trades["quote_qty"].resample(BUCKET).sum()
    agg["trade_count"] = trades["price"].resample(BUCKET).count()
    return agg


def compute_rolling_features(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.sort_index()

    agg["close"] = agg["close"].ffill()
    agg["open"] = agg["open"].fillna(agg["close"])
    agg["high"] = agg["high"].fillna(agg["close"])
    agg["low"] = agg["low"].fillna(agg["close"])
    agg[["volume", "notional", "trade_count"]] = agg[
        ["volume", "notional", "trade_count"]
    ].fillna(0.0)

    agg["log_return"] = np.log(agg["close"] / agg["close"].shift(1))
    agg["realized_vol"] = agg["log_return"].rolling(ROLL_WINDOW).std()
    agg["trade_intensity"] = agg["trade_count"].rolling(ROLL_WINDOW).mean()
    agg["avg_trade_size"] = (
        agg["notional"].rolling(ROLL_WINDOW).sum()
        / agg["trade_count"].rolling(ROLL_WINDOW).sum().replace(0, np.nan)
    )

    agg = agg.dropna(subset=["realized_vol", "trade_intensity"])
    return agg


def label_regimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    q_low, q_high = df["realized_vol"].quantile([0.33, 0.67])
    df["regime"] = np.select(
        [df["realized_vol"] <= q_low, df["realized_vol"] >= q_high],
        ["calm", "volatile"],
        default="normal",
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip_dir", required=True, help="Folder containing BTCUSDT-trades-*.zip files"
    )
    parser.add_argument(
        "--out", default="features.parquet", help="Output parquet file path"
    )
    args = parser.parse_args()

    zip_paths = sorted(glob.glob(os.path.join(args.zip_dir, "BTCUSDT-trades-*.zip")))
    if not zip_paths:
        raise SystemExit(f"No BTCUSDT-trades-*.zip files found in {args.zip_dir}")

    print(f"Found {len(zip_paths)} monthly files:")
    for p in zip_paths:
        print(f"  {os.path.basename(p)}")

    monthly_aggs = []
    total_raw = 0
    for p in zip_paths:
        print(f"Loading {os.path.basename(p)} ...")
        month_trades = load_trades_from_zip(p)
        total_raw += len(month_trades)
        print(f"  {len(month_trades):,} raw trades -> resampling to {BUCKET} buckets ...")
        month_agg = resample_trades(month_trades)
        monthly_aggs.append(month_agg)
        del month_trades

    print(f"Total raw trades processed across all months: {total_raw:,}")

    print("Concatenating monthly bucket tables ...")
    agg = pd.concat(monthly_aggs)
    del monthly_aggs

    print("Computing rolling features across full timeline ...")
    features = compute_rolling_features(agg)
    del agg

    print("Labeling volatility regimes ...")
    features = label_regimes(features)

    print(features["regime"].value_counts())
    print(features[["close", "realized_vol", "trade_intensity", "avg_trade_size", "regime"]].tail())

    features.to_parquet(args.out)
    print(f"Saved {len(features):,} rows to {args.out}")


if __name__ == "__main__":
    main()
