import numpy as np

from x.bpnn import NeuralNetwork


def test_nn():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    net = NeuralNetwork([2, 3, 1])
    net.train(X, y, epochs=1000, lr=0.1, momentum=0.9)
    net.test(X, y)
