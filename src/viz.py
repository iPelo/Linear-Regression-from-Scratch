import os
import matplotlib.pyplot as plt
import numpy as np

def save_pred_plot(x, y, m, b, title, path_png):
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.figure()
    plt.scatter(x, y)  # test points
    x_min, x_max = int(np.min(x)), int(np.max(x))
    x_line = np.arange(x_min, x_max + 1)
    plt.plot(x_line, m * x_line + b)
    plt.xlabel("G2 (second period grade)")
    plt.ylabel("G3 (final grade)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=200, bbox_inches="tight")
    plt.close()
    return path_png

def save_loss_curve(losses, epochs, path_png):
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.title(f"Loss Curve — epochs={epochs}")
    plt.tight_layout()
    plt.savefig(path_png, dpi=200, bbox_inches="tight")
    plt.close()
    return path_png