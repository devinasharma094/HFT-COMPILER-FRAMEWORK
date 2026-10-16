#!/usr/bin/env python3

import sys


def check_latency(
    baseline_p99,
    candidate_p99,
    baseline_p999,
    candidate_p999
):
    p99_ok = candidate_p99 <= baseline_p99
    p999_ok = candidate_p999 <= baseline_p999

    print(f"BASELINE  p99={baseline_p99:.1f} ns  p99.9={baseline_p999:.1f} ns")
    print(f"CANDIDATE p99={candidate_p99:.1f} ns  p99.9={candidate_p999:.1f} ns")

    print(
        f"LATENCY GATE: p99 {'PASS' if p99_ok else 'FAIL'}"
    )
    print(
        f"LATENCY GATE: p99.9 {'PASS' if p999_ok else 'FAIL'}"
    )

    if not (p99_ok and p999_ok):
        raise RuntimeError(
            "ADMISSION GATE: latency SLO failed"
        )

    print("ADMISSION GATE: latency SLO PASS")
    return True


if __name__ == "__main__":
    check_latency(
        float(sys.argv[1]),
        float(sys.argv[2]),
        float(sys.argv[3]),
        float(sys.argv[4])
    )
