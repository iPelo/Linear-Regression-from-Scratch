import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import title

dataM = pd.read_csv('/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/data/student-mat.csv', sep = ";")
dataP  = pd.read_csv('/Users/r.ilay/Desktop/AI Portfolio/Student_Regression/data/student-por.csv', sep = ";")

def loss_function(m,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].G2
        y = points.iloc[i].G3
        total_error += ( y - ( m * x + b )) ** 2
    return total_error / float(len(points))

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

m = 0
b = 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    if i % 50 ==0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, dataM, L)

print(m, b)

plt.scatter(dataM.G2, dataM.G3, color = "black")
x_min = int(dataM["G2"].min())
x_max = int(dataM["G2"].max())
x_line = list(range(x_min, x_max + 1))
plt.plot(x_line, [m * x + b for x in x_line], color="red")
plt.show()