def dual_drift_trigger(covariate_triggered, calibration_triggered):
    """
    Trigger regeneration if either drift detector fires.
    """

    triggered = (
        covariate_triggered
        or calibration_triggered
    )

    return {
        "covariate_triggered": bool(covariate_triggered),
        "calibration_triggered": bool(calibration_triggered),
        "regeneration_required": bool(triggered),
    }

if __name__ == "__main__":
    print("Dual drift trigger module ready")
