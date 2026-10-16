import subprocess
import sys
from pathlib import Path

from dual_drift_trigger import dual_drift_trigger


def regenerate(regime):
    cmd = [
        sys.executable,
        "generate_specialized_code.py",
        "--model", "model_big.json",
        "--vec", "vec_big.json",
        "--regenerate", regime,
    ]

    print("\n=== REGENERATION START ===")
    print("Regime:", regime)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError("Regeneration failed")

    print("=== REGENERATION SUCCESS ===")


def run_admission_gate():
    cmd = [
        sys.executable,
        "admission_gate.py",
        "--features", "features.parquet",
    ]

    print("\n=== ADMISSION GATE START ===")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("=== ADMISSION GATE: ACCEPT ===")
        return True

    print("=== ADMISSION GATE: REJECT ===")
    return False


def main():
    # Test input for the end-to-end integration.
    covariate_triggered = True
    calibration_triggered = False

    result = dual_drift_trigger(
        covariate_triggered,
        calibration_triggered,
    )

    print("=== DUAL DRIFT RESULT ===")
    print(result)

    if result["regeneration_required"]:
        regenerate("normal")
        accepted = run_admission_gate()

        if accepted:
            print("=== CANDIDATE APPROVED FOR SWAP ===")

            if not Path("provenance.json").exists():
                raise RuntimeError("Provenance record missing — swap blocked")

            print("=== PROVENANCE VERIFIED ===")
            subprocess.run(["./atomic_swap"], check=True)
        else:
            print("=== CANDIDATE REJECTED — KEEP CURRENT MODEL ===")
    else:
        print("No regeneration required.")


if __name__ == "__main__":
    main()
