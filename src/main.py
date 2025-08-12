import os, json
from datetime import datetime

from data import load_student_csv, train_test_split_df, extract_xy
from models import LinearRegressor
from viz import save_pred_plot, save_loss_curve

# === configurable bits ===
DATA_PATH = "/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/data/student-mat.csv"  # Math first
OUTPUTS  = "/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/outputs"
LR       = 1e-4
EPOCHS   = 1000
SEED     = 42

def run_one(csv_path, outputs_dir, lr=LR, epochs=EPOCHS, seed=SEED, verbose_every=100):
    df = load_student_csv(csv_path)
    train_df, test_df = train_test_split_df(df, test_size=0.2, seed=seed)

    x_tr, y_tr = extract_xy(train_df, "G2", "G3")
    x_te, y_te = extract_xy(test_df, "G2", "G3")

    # --- Gradient Descent ---
    gd = LinearRegressor(lr=lr, epochs=epochs).fit_gd(x_tr, y_tr, verbose_every=verbose_every)
    train_mse = gd.train_mse(x_tr, y_tr)
    test_mse  = gd.test_mse(x_te, y_te)
    test_r2   = gd.test_r2(x_te, y_te)

    title = f"GD — epochs={epochs}, m={gd.m_:.4f}, b={gd.b_:.4f}"
    pred_plot = os.path.join(outputs_dir, f"gd_reg_e{epochs}_m{gd.m_:.4f}_b{gd.b_:.4f}.png")
    save_pred_plot(x_te, y_te, gd.m_, gd.b_, title, pred_plot)

    loss_plot = os.path.join(outputs_dir, f"gd_loss_e{epochs}_L{lr:.0e}.png")
    save_loss_curve(gd.loss_curve_, epochs, loss_plot)

    # --- Normal Equation (sanity) ---
    ne = LinearRegressor().fit_normal(x_tr, y_tr)
    ne_test_mse = ne.test_mse(x_te, y_te)
    ne_test_r2  = ne.test_r2(x_te, y_te)

    # --- Save metrics JSON ---
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": "math_G2_to_G3",
        "method": "gd",
        "epochs": epochs,
        "learning_rate": lr,
        "m": gd.m_, "b": gd.b_,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "test_r2": test_r2,
        "ne_m": ne.m_, "ne_b": ne.b_,
        "ne_test_mse": ne_test_mse,
        "ne_test_r2": ne_test_r2,
        "pred_plot": pred_plot,
        "loss_plot": loss_plot,
        "seed": seed,
    }
    os.makedirs(outputs_dir, exist_ok=True)
    metrics_path = os.path.join(outputs_dir, f"metrics_gd_e{epochs}_L{lr:.0e}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved:\n- {pred_plot}\n- {loss_plot}\n- {metrics_path}")

if __name__ == "__main__":
    run_one(DATA_PATH, OUTPUTS)