import numpy as np

def mse(y_true, y_pred) -> float:
    return float(np.mean((y_true - y_pred)**2))

def r2(y_true, y_pred) -> float:
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    return float('nan') if ss_tot == 0 else 1.0 - ss_res/ss_tot