from sklearn.datasets import make_blobs, make_moons
import numpy as np

def get_dataset(name):
    if name == "Blobs":
        X, y = make_blobs(n_samples=200, centers=2, random_state=42)
    elif name == "Moons":
        X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    elif name == "Linear":
        X = np.random.rand(200, 1) * 10
        y = 2 * X + 1 + np.random.randn(200, 1)
        return X, y.ravel()
    return X[:, 0:1], y
