import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from datetime import datetime

dataM = pd.read_csv('/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/data/student-mat.csv', sep=";")
dataP = pd.read_csv('/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/data/student-por.csv', sep=";")

shuffled_dataM = dataM.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(len(shuffled_dataM) * 0.8)
train_data = shuffled_dataM.iloc[:split_index]
test_data = shuffled_dataM.iloc[split_index:]


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].G2
        y = points.iloc[i].G3
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def r2_score(m, b, points):
    y = points["G3"].values
    x = points["G2"].values
    y_pred = m * x + b

    ss_res = ((y - y_pred) ** 2).sum()
    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum()

    if ss_tot == 0:
        return float('nan')
    return 1 - ss_res / ss_tot


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].G2
        y = points.iloc[i].G3

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b


def normal_equation(points):
    # Build design matrix with bias: X = [x, 1]
    x = points["G2"].values.astype(float)
    y = points["G3"].values.astype(float)
    X = np.column_stack([x, np.ones_like(x)])
    # theta = (X^T X)^{-1} X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    theta = np.linalg.pinv(XtX) @ Xty
    m_hat = float(theta[0])
    b_hat = float(theta[1])
    return m_hat, b_hat


def save_plot(m, b, points, epochs, path=None):
    if path is None:
        outdir = "plots"
        os.makedirs(outdir, exist_ok=True)
        fname = f"reg_epoch_{epochs}_m_{m:.4f}_b_{b:.4f}.png"
        full_path = os.path.join(outdir, fname)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        full_path = path

    plt.scatter(points.G2, points.G3, color="black")
    x_min = int(points["G2"].min())
    x_max = int(points["G2"].max())
    x_line = list(range(x_min, x_max + 1))
    plt.plot(x_line, [m * x + b for x in x_line], color="red")
    plt.xlabel("G2 (second period grade)")
    plt.ylabel("G3 (final grade)")
    plt.title(f"Linear Regression — epochs={epochs}, m={m:.4f}, b={b:.4f}")
    plt.text(
        0.02, 0.98,
        f"Epochs: {epochs}\nm: {m:.4f}\nb: {b:.4f}",
        transform=plt.gca().transAxes,
        va='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
    )
    plt.savefig(full_path, dpi=200, bbox_inches="tight")
    plt.close()
    return full_path


def save_loss_curve(losses, epochs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.title(f"Loss Curve — epochs={epochs}")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


m = 0
b = 0
L = 0.0001
epochs = 1000

losses = []
outputs_base = "/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/outputs"

for i in range(epochs):
    m, b = gradient_descent(m, b, train_data, L)
    curr_train_loss = loss_function(m, b, train_data)
    losses.append(curr_train_loss)
    if i % 100 == 0:
        print(f"Epoch: {i}  Train MSE: {curr_train_loss:.6f}")

print(m, b)

train_loss = loss_function(m, b, train_data)
test_loss = loss_function(m, b, test_data)
print(f"Train MSE: {train_loss:.4f}")
print(f"Test MSE: {test_loss:.4f}")
test_r2 = r2_score(m, b, test_data)
print(f"Test R^2: {test_r2:.4f}")

# Closed-form solution (sanity check)
m_ne, b_ne = normal_equation(train_data)
test_loss_ne = loss_function(m_ne, b_ne, test_data)
test_r2_ne = r2_score(m_ne, b_ne, test_data)
print(f"[NE] Test MSE: {test_loss_ne:.4f}")
print(f"[NE] Test R^2: {test_r2_ne:.4f}")

# Save prediction plot (test set)
saved_path = save_plot(
    m, b, test_data, epochs,
    path=f"{outputs_base}/reg_epoch_{epochs}_m_{m:.4f}_b_{b:.4f}.png"
)
print(f"Saved plot to: {saved_path}")

# Save loss curve
loss_path = save_loss_curve(
    losses, epochs,
    path=f"{outputs_base}/loss_curve_e{epochs}_L{L:.0e}.png"
)
print(f"Saved loss curve to: {loss_path}")

# Save metrics JSON
metrics = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "method": "gd",
    "epochs": epochs,
    "learning_rate": L,
    "m": m,
    "b": b,
    "train_mse": train_loss,
    "test_mse": test_loss,
    "test_r2": test_r2,
    "ne_m": m_ne,
    "ne_b": b_ne,
    "ne_test_mse": test_loss_ne,
    "ne_test_r2": test_r2_ne,
    "plot_path": saved_path,
    "loss_curve_path": loss_path,
    "dataset": "math_G2_to_G3"
}
metrics_path = f"{outputs_base}/metrics_e{epochs}_L{L:.0e}.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics to: {metrics_path}")
