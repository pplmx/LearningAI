import numpy as np


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))


class Tanh:
    def __call__(self, x):
        return np.tanh(x)


class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.maximum(self.alpha * x, x)
