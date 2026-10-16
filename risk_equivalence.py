import subprocess
import tempfile
import os

REFERENCE_FILE = "risk_control.c"
CANDIDATE_FILE = "risk_control_bad.c"

TEST_CASES = [
    (50.0, 5.0),
    (95.0, 10.0),
    (50.0, 15.0),
    (-90.0, 5.0),
    (0.0, 10.0),
    (0.0, 11.0),
]


def run_risk_test(source_file):
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "risk_test")

        test_c = os.path.join(tmp, "test.c")

        with open(test_c, "w") as f:
            f.write("""
#include <stdio.h>

int risk_check(double position, double order_size);

int main(void)
{
""")

            for position, order_size in TEST_CASES:
                f.write(
                    f'    printf("%d\\n", '
                    f'risk_check({position}, {order_size}));\n'
                )

            f.write("""
    return 0;
}
""")

        subprocess.run(
            ["clang", source_file, test_c, "-o", exe],
            check=True,
            capture_output=True
        )

        result = subprocess.run(
            [exe],
            check=True,
            capture_output=True,
            text=True
        )

        return result.stdout.strip().splitlines()


def main():
    reference = run_risk_test(REFERENCE_FILE)

    # For now the candidate is the same protected risk implementation.
    candidate = run_risk_test(CANDIDATE_FILE)

    print("REFERENCE:", reference)
    print("CANDIDATE:", candidate)

    if reference != candidate:
        raise RuntimeError(
            "ADMISSION GATE: risk-limit outputs differ"
        )

    print("ADMISSION GATE: risk equivalence PASS")


if __name__ == "__main__":
    main()
