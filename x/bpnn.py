#!/usr/bin/env python

import numpy as np


class NeuralNetwork:
    def __init__(self, layers, lambd=0.1):
        """
        初始化神经网络
        参数:
        - layers: 网络层数列表,例如[2,4,1]表示输入层2个节点,隐层4个节点,输出层1个节点
        - lambd: L2正则化系数
        """
        self.layers = layers

        # 权重矩阵初始化为小的随机数
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

        # 偏置向量初始化为小的随机数
        self.biases = [np.random.randn(x, 1) for x in layers[1:]]

        self.lambd = lambd  # L2正则化系数

        # 初始化动量
        self.velocity = [np.zeros_like(w) for w in self.weights]

    def forward(self, x):
        """
        网络的前向传播
        输入x,返回网络的输出
        参数:
        - x: 输入数据
        返回:
        - 输出层的激活值
        """
        a = x
        for w, b in zip(self.weights, self.biases):
            # 权重矩阵点乘输入加偏置
            z = np.dot(w, a) + b
            # 通过激活函数
            a = self.sigmoid(z)
        return a

    def backward(self, x, y):
        """
        网络的反向传播
        输入x和y,计算损失,返回损失和每个参数的梯度
        参数:
        - x: 输入数据
        - y: 标签
        返回:
        - loss: 损失函数值
        - grads: 权重/偏置的参数梯度字典
        """
        grads = {}
        loss = 0
        a = x

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # 前向传播
            z = np.dot(w, a) + b
            a = self.sigmoid(z)

            # 计算损失
            if i == len(self.weights) - 1:
                loss += np.sum((y - a) ** 2)
            # 反向传播误差
            err = (
                (y - a)
                if i == len(self.weights) - 1
                else np.dot(self.weights[i + 1].T, err)
            )

            # 计算梯度
            dw = np.outer(a, err) + self.lambd * w  # 加入了L2正则化
            db = err

            grads["w" + str(i + 1)] = dw
            grads["b" + str(i + 1)] = db

        return loss, grads

    def sigmoid(self, x):
        # 激活函数
        return 1 / (1 + np.exp(-x))

    def train(self, X, y, epochs=100, lr=0.01, momentum=0.9):
        """
        训练网络,更新参数
        输入样本和标签,以及训练超参数,进行渐变下降训练网络
        参数:
        - X: 训练数据
        - y: 训练标签
        - epochs: 遍历数据集的次数
        - lr: 学习率
        - momentum: 动量因子
        """
        for i in range(epochs):
            loss, grads = self.backward(X, y)
            for k, v in grads.items():
                # 添加动量
                self.velocity[int(k[1:])] = (
                    momentum * self.velocity[int(k[1:])] - lr * v
                )
                self.weights[int(k[1:])] += self.velocity[int(k[1:])]
                self.biases[int(k[1:])] -= lr * v.sum(axis=0)

    def test(self, X, y):
        """测试模型"""
        out = self.forward(X)
        print(f"预测值:{out}\n")

        # 测试预测是否接近标签
        assert np.allclose(out, y, atol=1e-2)
