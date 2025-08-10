import pandas as pd
import matplotlib.pyplot as plt
import os

dataM = pd.read_csv('/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/data/student-mat.csv', sep = ";")
dataP  = pd.read_csv('/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/data/student-por.csv', sep = ";")

shuffled_dataM = dataM.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(len(shuffled_dataM) * 0.8)
train_data = shuffled_dataM.iloc[:split_index]
test_data = shuffled_dataM.iloc[split_index:]

def loss_function(m,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].G2
        y = points.iloc[i].G3
        total_error += ( y - ( m * x + b )) ** 2
    return total_error / float(len(points))

def r2_score(m,b,points):
    y = points["G3"].values
    x = points["G2"].values
    y_pred = m * x +b

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

        m_gradient += -(2/n) * x * (y - (m_now * x +  b_now))
        b_gradient += -(2/n) * (y - (m_now * x +  b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b

def save_plot(m, b, points, epochs, path=None):
    if path is None:
        outdir =  "plots"
        os.makedirs(outdir, exist_ok=True)
        fname = f"reg_epoch_{epochs}_m_{m:.4f}_b_{b:.4f}.png"
        full_path = os.path.join(outdir, fname)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        full_path = path

    plt.scatter(points.G2, points.G3, color = "black")
    x_min = int(points["G2"].min())
    x_max = int(points["G2"].max())
    x_line = list(range(x_min, x_max + 1))
    plt.plot(x_line, [m * x + b for x in x_line], color="red")
    plt.text(
        0.02, 0.98,
        f"Epochs: {epochs}\nm: {m:.4f}\nb: {b:.4f}",
        transform=plt.gca().transAxes,
        va='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
    )
    plt.savefig(full_path)
    plt.close()

m = 0
b = 0
L = 0.0001
epochs = 50

for i in range(epochs):
    if i % 50 ==0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, train_data, L)

print(m, b)

train_loss = loss_function(m, b, train_data)
test_loss = loss_function(m, b, test_data)
print(f"Train MSE: {train_loss:.4f}")
print(f"Test MSE: {test_loss:.4f}")
test_r2 = r2_score(m, b, test_data)
print(f"Test R^2: {test_r2:.4f}")

#save_plot(
    #m, b, dataM, epochs,
  #  path=f"/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/outputs/reg_epoch_{epochs}_m_{m:.4f}_b_{b:.4f}.png"
#)