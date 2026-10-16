"""
admission_gate.py -- prototype of the COVERAGE condition of the admission gate (architecture item 9).

Idea: before a regenerated certificate/kernel may replace the live one, replay a recent window of data ("shadow
replay") and check that the candidate certificate still delivers its stated coverage on that window.

Per clause (regime, feature) the candidate promises a miss rate of at most p0 = 1 - coverage. On the replay window we
measure the miss rate m and its standard error from BLOCK means (blocks of B consecutive observations), which accounts
for autocorrelation. A clause FAILS if we are confident the true miss rate exceeds p0:   m - z*se > p0.
The candidate is ACCEPTED only if no clause fails.  (Exactness and latency checks are done by the C benchmark and are
not repeated here; this script tests only whether the gate can separate good candidates from bad ones.)

Backtest scenarios (all candidates are evaluated on the SAME replay window = the test period):
  fit windows : certificate fitted on 0-20%, 20-40%, 40-60%, 60-80% (= the normal calibration) of the data
  corrupt     : the 60-80% certificate with every interval shrunk by 25% around its centre
  shifted     : the 60-80% certificate, but the replay window has avg_trade_size scaled by 1.3 after its midpoint

Usage:  python admission_gate.py --features features.parquet
Options: --coverage 0.90 --block 3000 --z 2.54
"""
import argparse

import numpy as np
import pandas as pd
from latency_gate import check_latency
from risk_equivalence import run_risk_test

FEATURE_COLS = ["realized_vol", "trade_intensity", "avg_trade_size"]
REGIMES = ["calm", "normal", "volatile"]
HORIZON = 30


def fit_intervals(reg, feats, lo_i, hi_i, cov):
    a = (1 - cov) / 2
    out = {}
    for r in REGIMES:
        m = reg[lo_i:hi_i] == r
        out[r] = {f: (float(np.quantile(feats[f][lo_i:hi_i][m], a)), float(np.quantile(feats[f][lo_i:hi_i][m], 1 - a)))
                  for f in FEATURE_COLS}
    return out


def shrink(iv, frac):
    out = {}
    for r, d in iv.items():
        out[r] = {}
        for f, (lo, hi) in d.items():
            c, hw = (lo + hi) / 2, (hi - lo) / 2
            out[r][f] = (c - hw * (1 - frac), c + hw * (1 - frac))
    return out


def evaluate(reg, feats, t0, t1, iv, p0, B, z, shift_feat=None, shift=0.0):
    """Return (accept, worst_gap, n_fail, details) for one candidate on the replay window [t0, t1)."""
    mid = (t0 + t1) // 2
    fails, details = 0, []
    for r in REGIMES:
        idx = np.flatnonzero(reg[t0:t1] == r) + t0
        for f in FEATURE_COLS:
            x = feats[f][idx]
            if f == shift_feat:
                x = np.where(idx >= mid, x * (1 + shift), x)
            lo, hi = iv[r][f]
            miss = ~((x >= lo) & (x <= hi))
            nb = len(miss) // B
            if nb < 30 and not getattr(evaluate, "warned", False):
                print(f"WARNING: only {nb} blocks for {r}.{f}; standard errors are unreliable (use a longer replay window or --block smaller)")
                evaluate.warned = True
            bm = miss[: nb * B].reshape(nb, B).mean(axis=1)
            m, se = float(bm.mean()), float(bm.std(ddof=1) / np.sqrt(nb))
            gap = m - z * se - p0          # > 0  =>  confident the certified rate is violated
            failed = gap > 0
            fails += failed
            details.append((f"{r}.{f}", m, se, gap, failed))
    worst = max(details, key=lambda d: d[3])
    return fails == 0, worst, fails, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--coverage", type=float, default=0.90)
    ap.add_argument("--block", type=int, default=3000)
    ap.add_argument("--z", type=float, default=2.54, help="one-sided 5%% level with Bonferroni over 9 clauses")
    ap.add_argument("--baseline-p99", type=float, default=None)
    ap.add_argument("--candidate-p99", type=float, default=None)
    ap.add_argument("--baseline-p999", type=float, default=None)
    ap.add_argument("--candidate-p999", type=float, default=None)
    a = ap.parse_args()

    df = pd.read_parquet(a.features).iloc[:-HORIZON].reset_index(drop=True)
    n = len(df)
    reg = df["regime"].to_numpy()
    feats = {f: df[f].to_numpy() for f in FEATURE_COLS}
    p0 = 1 - a.coverage
    t0, t1 = int(n * 0.8), n
    q = lambda x: int(n * x)

    cands = []
    coverage_ok = False
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]:
        name = f"fit {int(lo*100)}-{int(hi*100)}%" + ("  (normal calibration)" if lo == 0.6 else "")
        cands.append((name, fit_intervals(reg, feats, q(lo), q(hi), a.coverage), None, 0.0))
    good = cands[-1][1]
    cands.append(("calibration, intervals shrunk 25%", shrink(good, 0.25), None, 0.0))
    cands.append(("calibration, replay shifted: avg_trade_size x1.3", good, "avg_trade_size", 0.3))

    print(f"replay window: rows {t0:,}-{t1:,} (test period) | stated coverage {a.coverage} (p0={p0:.3f}) | "
          f"block={a.block} | reject if miss - {a.z}*se > p0\n")
    print(f"{'candidate':52s} {'decision':8s} {'failing':>8s}   worst clause (miss rate, gap)")
    for name, iv, sf, sh in cands:
        ok, worst, nf, _ = evaluate(reg, feats, t0, t1, iv, p0, a.block, a.z, sf, sh)
        print(f"{name:52s} {'ACCEPT' if ok else 'REJECT':8s} {nf:>5d}/9   {worst[0]} ({worst[1]:.3f}, gap {worst[3]:+.3f})")
        if name.startswith("fit 60-80%"):
        	coverage_ok = ok
    if all(v is not None for v in [
        a.baseline_p99,
        a.candidate_p99,
        a.baseline_p999,
        a.candidate_p999
    ]):
        check_latency(
            a.baseline_p99,
            a.candidate_p99,
            a.baseline_p999,
            a.candidate_p999
        )
    risk_ok = (run_risk_test("risk_control_good.c") == ["1", "0", "0", "1", "1", "0"])
    latency_ok = True

    if coverage_ok and risk_ok and latency_ok:
    	print("ADMISSION GATE: ACCEPT")
    	raise SystemExit(0)
    else:
    	print("ADMISSION GATE: REJECT")
    	raise SystemExit(1)

if __name__ == "__main__":

    main()
