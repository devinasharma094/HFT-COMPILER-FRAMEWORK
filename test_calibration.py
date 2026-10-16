import numpy as np
from calibration_trigger import calibration_trigger


data = np.load("calibration_test_data.npz")

y_true = data["y_true"]
y_prob = data["y_prob"]

result = calibration_trigger(y_true, y_prob)

print("REAL HFT TEST DATA:")
print(result)
