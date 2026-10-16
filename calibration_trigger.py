import numpy as np


def calibration_trigger(
    y_true,
    y_prob,
    threshold=0.10
):
    """
    Detect degradation in prediction calibration.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    predicted = (y_prob >= 0.5).astype(int)

    accuracy = np.mean(predicted == y_true)

    confidence = np.mean(
        np.maximum(y_prob, 1.0 - y_prob)
    )

    calibration_error = abs(confidence - accuracy)

    triggered = calibration_error > threshold

    return {
        "accuracy": float(accuracy),
        "confidence": float(confidence),
        "calibration_error": float(calibration_error),
        "triggered": bool(triggered),
    }


if __name__ == "__main__":
    y_true = np.array([
        1, 1, 0, 0, 1,
        0, 1, 0, 1, 0
    ])

    y_prob = np.array([
    0.90, 0.85, 0.20, 0.15, 0.80,
    0.10, 0.90, 0.20, 0.85, 0.15
])

    result = calibration_trigger(y_true, y_prob)

    print(result)
