import os, json
from datetime import datetime
import argparse

from data import load_student_csv, train_test_split_df, extract_xy
from models import LinearRegressor
from viz import save_pred_plot, save_loss_curve

# === defaults (adjust to your machine if needed) ===
DATA_DIR = "/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/data"
DATA_PATH_MATH = os.path.join(DATA_DIR, "student-mat.csv")
DATA_PATH_PORT = os.path.join(DATA_DIR, "student-por.csv")
OUTPUTS        = "/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/outputs"
LR             = 1e-4
EPOCHS         = 1000
SEED           = 42

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

    dataset_label = os.path.splitext(os.path.basename(csv_path))[0]

    title = f"{dataset_label} — GD: epochs={epochs}, m={gd.m_:.4f}, b={gd.b_:.4f}"
    pred_plot = os.path.join(outputs_dir, f"{dataset_label}_gd_reg_e{epochs}_m{gd.m_:.4f}_b{gd.b_:.4f}.png")
    save_pred_plot(x_te, y_te, gd.m_, gd.b_, title, pred_plot)

    loss_plot = os.path.join(outputs_dir, f"{dataset_label}_gd_loss_e{epochs}_L{lr:.0e}.png")
    save_loss_curve(gd.loss_curve_, epochs, loss_plot)

    # --- Normal Equation ---
    ne = LinearRegressor().fit_normal(x_tr, y_tr)
    ne_test_mse = ne.test_mse(x_te, y_te)
    ne_test_r2  = ne.test_r2(x_te, y_te)

    # --- Save metrics JSON ---
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_label,
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
    metrics_path = os.path.join(outputs_dir, f"{dataset_label}_metrics_gd_e{epochs}_L{lr:.0e}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{dataset_label}] Saved:\n- {pred_plot}\n- {loss_plot}\n- {metrics_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run linear regression (GD) on the UCI student datasets.")
    parser.add_argument("--dataset", choices=["math", "port", "both"], default="math",
                        help="Which dataset to run: math (student-mat), port (student-por), or both.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate for gradient descent.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for the train/test split.")
    parser.add_argument("--outputs", default=OUTPUTS, help="Directory to save plots and metrics.")
    parser.add_argument("--verbose-every", type=int, default=100,
                        help="Print training MSE every N epochs (0 to disable).")
    parser.add_argument("--mat-path", default=DATA_PATH_MATH, help="Path to student-mat.csv")
    parser.add_argument("--por-path", default=DATA_PATH_PORT, help="Path to student-por.csv")
    return parser.parse_args()

def main():
    # Interactive choice for PyCharm
    print("Select dataset to run:")
    print("1 - Mathematics (student-mat.csv)")
    print("2 - Portuguese (student-por.csv)")
    print("3 - Both")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        dataset_choice = "math"
    elif choice == "2":
        dataset_choice = "port"
    elif choice == "3":
        dataset_choice = "both"
    else:
        print("Invalid choice, defaulting to Mathematics dataset.")
        dataset_choice = "math"

    args = parse_args()
    args.dataset = dataset_choice  # Override CLI dataset with interactive choice

    if args.dataset in ("math", "both"):
        run_one(args.mat_path, args.outputs, lr=args.lr, epochs=args.epochs,
                seed=args.seed, verbose_every=args.verbose_every)

    if args.dataset in ("port", "both"):
        run_one(args.por_path, args.outputs, lr=args.lr, epochs=args.epochs,
                seed=args.seed, verbose_every=args.verbose_every)

if __name__ == "__main__":
    main()