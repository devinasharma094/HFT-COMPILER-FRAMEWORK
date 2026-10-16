"""
drift_monitor.py (v2) -- covariate drift trigger prototype (architecture item 7, first half).

Each certified interval clause (regime, feature) promises a miss rate p0 = 1 - coverage for rows of that regime.
We monitor the stream of per-row "miss" indicators (feature outside its interval) in time order.

Two monitors are run side by side so the effect of dependence is visible:
  iid   : per-row Bernoulli CUSUM (a repeated SPRT). Assumes independent rows -> expected to false-alarm on
          autocorrelated market data.
  batch : batch-means CUSUM. Consecutive observations of the clause are grouped into blocks of B; each block's miss
          rate is standardized with sigma estimated on the FIRST HALF of the calibration period (this captures the
          real block-to-block variability, including autocorrelation), then a one-sided CUSUM
              S_j = max(0, S_{j-1} + z_j - k),  alarm when S_j >= h,  reset after alarm
          is applied. The 'VIF' column is the variance-inflation factor = (sigma_hat / binomial sigma)^2.

Evaluation periods (chronological split used everywhere else): train 60% | calib 20% | test 20%.
calibA = first half of calib (sigma fitted here, in-sample), calibB = second half of calib (held out).
Alarms in calibB are FALSE alarms by construction (certificate fitted on calib).

Usage:
    python drift_monitor.py --features features.parquet --vec vec.json
Options: --block 3000 --k 0.5 --h 8 --delta 0.05 --h_iid 13.8 --shifts 0.1 0.25 0.5
"""
import argparse
import json
import math

import numpy as np
import pandas as pd

FEATURE_COLS = ["realized_vol", "trade_intensity", "avg_trade_size"]
REGIMES = ["calm", "normal", "volatile"]
HORIZON = 30


def iid_cusum(miss, inc_miss, inc_hit, h):
    s, out = 0.0, []
    for i, m in enumerate(miss.tolist()):
        s += inc_miss if m else inc_hit
        if s < 0.0:
            s = 0.0
        elif s >= h:
            out.append(i)
            s = 0.0
    return np.array(out, dtype=int)


def block_cusum(z, k, h):
    s, out = 0.0, []
    for j, v in enumerate(z.tolist()):
        s += v - k
        if s < 0.0:
            s = 0.0
        elif s >= h:
            out.append(j)
            s = 0.0
    return np.array(out, dtype=int)


def block_stats(miss, idx, B):
    """Block miss rates and the global row index at which each block ends."""
    nb = len(miss) // B
    bm = miss[: nb * B].reshape(nb, B).mean(axis=1)
    ends = idx[(np.arange(nb) + 1) * B - 1]
    return bm, ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--vec", required=True)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--h_iid", type=float, default=13.8)
    ap.add_argument("--block", type=int, default=3000, help="observations per block (per clause)")
    ap.add_argument("--k", type=float, default=0.5, help="CUSUM reference value in sigma units")
    ap.add_argument("--h", type=float, default=8.0, help="CUSUM alarm threshold in sigma units")
    ap.add_argument("--shifts", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    a = ap.parse_args()

    df = pd.read_parquet(a.features).iloc[:-HORIZON].reset_index(drop=True)
    n = len(df)
    c0, c1 = int(n * 0.6), int(n * 0.8)
    cmid = (c0 + c1) // 2
    periods = {"train": (0, c0), "calibA": (c0, cmid), "calibB": (cmid, c1), "test": (c1, n)}
    vec = json.load(open(a.vec))
    cov = vec["coverage_level"]
    p0 = 1.0 - cov
    p1 = min(p0 + a.delta, 0.99)
    inc_miss, inc_hit = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    B = a.block
    sig_bin = math.sqrt(p0 * (1 - p0) / B)
    print(f"coverage {cov} p0={p0:.3f} | iid CUSUM h={a.h_iid} | batch CUSUM block={B} k={a.k} h={a.h}")
    print(f"rows {n:,}; " + ", ".join(f"{k}={v[1]-v[0]:,}" for k, v in periods.items()))

    regime = df["regime"].to_numpy()
    clauses = [(r, f) for r in REGIMES for f in FEATURE_COLS if f in vec["feature_intervals"].get(r, {})]
    fitted = {}

    print(f"\n== 1. Miss rate and alarms per period: rate  iidAlarms/batchAlarms  (certified {p0:.3f}) ==")
    print(f"{'clause':26s} {'VIF':>6s} " + " ".join(f"{k:>17s}" for k in periods))
    for r, f in clauses:
        iv = vec["feature_intervals"][r][f]
        idx = np.flatnonzero(regime == r)
        x = df[f].to_numpy()[idx]
        miss = ~((x >= iv["lower_bound"]) & (x <= iv["upper_bound"]))
        iid_al = iid_cusum(miss, inc_miss, inc_hit, a.h_iid)
        bm, ends = block_stats(miss, idx, B)
        fit = (ends >= c0) & (ends < cmid)
        sig = max(float(np.std(bm[fit], ddof=1)), sig_bin) if fit.sum() > 3 else sig_bin
        fitted[(r, f)] = sig
        z = (bm - p0) / sig
        bat_al = block_cusum(z, a.k, a.h)
        cells = []
        for k_, (lo_i, hi_i) in periods.items():
            sel = (idx >= lo_i) & (idx < hi_i)
            rate = miss[sel].mean() if sel.any() else float("nan")
            ni = int(((idx[iid_al] >= lo_i) & (idx[iid_al] < hi_i)).sum()) if len(iid_al) else 0
            nbt = int(((ends[bat_al] >= lo_i) & (ends[bat_al] < hi_i)).sum()) if len(bat_al) else 0
            cells.append(f"{rate:5.3f} {ni:5d}/{nbt:<3d}")
        print(f"{r + '.' + f:26s} {(sig / sig_bin) ** 2:6.1f} " + " ".join(f"{c:>17s}" for c in cells))

    t0, t1 = periods["test"]
    mid = (t0 + t1) // 2
    print(f"\n== 2. Batch monitor: shift injected into TEST at its midpoint (feature x (1+s)). Delay in rows (~seconds). ==")
    print("   '*' = the unshifted stream also alarmed by then, so the alarm is not attributable to the shift")
    print(f"{'clause':26s} " + " ".join(f"s={s:<5g}".rjust(14) for s in a.shifts) + "   false alarms in test before shift")
    for r, f in clauses:
        iv = vec["feature_intervals"][r][f]
        idx_all = np.flatnonzero(regime == r)
        idx = idx_all[(idx_all >= t0) & (idx_all < t1)]
        x0 = df[f].to_numpy()[idx]
        sig = fitted[(r, f)]

        def run(x):
            miss = ~((x >= iv["lower_bound"]) & (x <= iv["upper_bound"]))
            bm, ends = block_stats(miss, idx, B)
            return ends, block_cusum((bm - p0) / sig, a.k, a.h)

        ends0, al0 = run(x0)
        pre_fa = int((ends0[al0] < mid).sum()) if len(al0) else 0
        base_after = ends0[al0][ends0[al0] >= mid] if len(al0) else np.array([])
        row = []
        for s in a.shifts:
            ends, al = run(np.where(idx >= mid, x0 * (1 + s), x0))
            post = ends[al][ends[al] >= mid] if len(al) else np.array([])
            if len(post):
                d = int(post[0] - mid)
                mark = "*" if len(base_after) and base_after[0] <= post[0] else ""
                row.append(f"{d}{mark}")
            else:
                row.append("not detected")
        print(f"{r + '.' + f:26s} " + " ".join(c.rjust(14) for c in row) + f"   {pre_fa}")


if __name__ == "__main__":
    main()
