import numpy as np

class LinearRegressor:
    def __init__(self, lr=1e-4, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.m_ = 0.0
        self.b_ = 0.0
        self.loss_curve_ = []

    @staticmethod
    def _loss(m, b, x, y):
        return float(np.mean((y - (m * x + b))**2))

    @staticmethod
    def _r2(m, b, x, y):
        y_pred = m * x + b
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float('nan') if ss_tot == 0 else 1.0 - ss_res/ss_tot

    def fit_gd(self, x_train, y_train, verbose_every=100):
        m, b = 0.0, 0.0
        n = x_train.shape[0]
        self.loss_curve_.clear()

        for i in range(self.epochs):
            y_hat = m * x_train + b
            m_grad = -(2.0/n) * np.sum(x_train * (y_train - y_hat))
            b_grad = -(2.0/n) * np.sum(y_train - y_hat)
            m -= self.lr * m_grad
            b -= self.lr * b_grad

            loss = self._loss(m, b, x_train, y_train)
            self.loss_curve_.append(loss)
            if verbose_every and (i % verbose_every == 0):
                print(f"Epoch {i}: Train MSE = {loss:.6f}")

        self.m_, self.b_ = float(m), float(b)
        return self

    def fit_normal(self, x_train, y_train):
        X = np.column_stack([x_train, np.ones_like(x_train)])
        theta = np.linalg.pinv(X.T @ X) @ (X.T @ y_train)
        self.m_, self.b_ = float(theta[0]), float(theta[1])
        self.loss_curve_.clear()
        return self

    def predict(self, x):
        return self.m_ * x + self.b_

    def train_mse(self, x_train, y_train):
        return self._loss(self.m_, self.b_, x_train, y_train)

    def test_mse(self, x_test, y_test):
        return self._loss(self.m_, self.b_, x_test, y_test)

    def test_r2(self, x_test, y_test):
        return self._r2(self.m_, self.b_, x_test, y_test)