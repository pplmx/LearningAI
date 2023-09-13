import numpy as np


class Loss:
    def __call__(self, pred, label):
        raise NotImplementedError

    def grad(self, pred, label):
        raise NotImplementedError


class MSELoss(Loss):
    def __call__(self, pred, label):
        return np.mean((pred - label) ** 2)

    def grad(self, pred, label):
        return 2 * (pred - label)


class CrossEntropyLoss(Loss):
    def __call__(self, pred, label):
        return -np.sum(label * np.log(pred))

    def grad(self, pred, label):
        return pred - label


class L1Loss(Loss):
    def __call__(self, pred, label):
        return np.sum(np.abs(pred - label))

    def grad(self, pred, label):
        return np.sign(pred - label)
