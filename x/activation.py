import numpy as np


class Activation:
    def __call__(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError


class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def grad(self, x):
        return self(x) * (1 - self(x))


class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)

    def grad(self, x):
        return 1 - self(x) ** 2


class ReLU(Activation):
    def __call__(self, x):
        return np.maximum(0, x)

    def grad(self, x):
        return 1 if x > 0 else 0


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.maximum(self.alpha * x, x)

    def grad(self, x):
        return 1 if x > 0 else self.alpha
