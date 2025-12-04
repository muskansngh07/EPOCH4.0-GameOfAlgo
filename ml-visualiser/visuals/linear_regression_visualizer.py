import numpy as np
import matplotlib.pyplot as plt
from .base_visualizer import BaseVisualizer

class LinearRegressionVisualizer(BaseVisualizer):
    def __init__(self, X, y, lr=0.01):
        self.X = X
        self.y = y
        self.lr = lr
        self.w = np.random.randn()
        self.b = np.random.randn()
        self.losses = []
        self.iteration = 0

    def step(self):
        y_pred = self.w * self.X + self.b
        error = y_pred - self.y.reshape(-1, 1)
        dw = (2/len(self.X)) * np.sum(error * self.X)
        db = (2/len(self.X)) * np.sum(error)
        self.w -= self.lr * dw
        self.b -= self.lr * db

        loss = np.mean(error ** 2)
        self.losses.append(loss)
        self.iteration += 1

    def plot(self, ax):
        ax.scatter(self.X, self.y, c='blue')
        x_line = np.linspace(min(self.X), max(self.X), 100)
        y_line = self.w * x_line + self.b
        ax.plot(x_line, y_line, linestyle='-', linewidth=2)

        if len(self.losses) > 0:
            mse = self.losses[-1]
            ax.set_title(f"Iteration: {self.iteration}, MSE: {mse:.3f}")
        else:
            ax.set_title("Iteration: 0, MSE: N/A")
