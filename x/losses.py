import numpy as np


class MSELoss:
    def __call__(self, pred, label):
        return np.mean((pred - label) ** 2)


class CrossEntropyLoss:
    def __call__(self, pred, label):
        return -np.sum(label * np.log(pred))


class L1Loss:
    def __call__(self, pred, label):
        return np.sum(np.abs(pred - label))
