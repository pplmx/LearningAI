#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import numpy as np

from activation import Sigmoid
from losses import MSELoss


class NeuralNetwork:
    def __init__(
        self, layers: List[int], lr: float = 0.01, momentum: float = 0.9
    ) -> None:
        """
        构造函数
        layers: 网络层数列表 - 例如[2,4,1]表示输入层2个节点,隐层4个节点,输出层1个节点
        lr: 学习率 - 梯度下降更新权重时的步长
        momentum: 动量因子 - 用于给梯度附加动量
        """
        self.layers = layers
        self.lr = lr
        self.momentum = momentum

        # 初始化权重和偏置
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.zeros(y) for y in layers[1:]]

        self.activation = Sigmoid()
        self.loss_fn = MSELoss()

    def inference(self, inputs: np.ndarray) -> np.ndarray:
        """前向传播,返回输出"""
        x = inputs
        for w, b in zip(self.weights, self.biases):
            # x * w + b
            x = np.dot(x, w) + b
            # 激活函数
            x = self.activation(x)
        return x

    def backprop(
        self, inputs: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """反向传播,返回损失和梯度"""
        # 前向传播
        x = inputs
        for w, b in zip(self.weights, self.biases):
            x = np.dot(x, w) + b
            x = self.activation(x)
        pred = x

        # 计算损失
        loss = self.loss_fn(pred, labels)

        # 反向传播
        grads = {}
        delta = (pred - labels) * self.activation.grad(x)
        for i in reversed(range(len(self.weights))):
            grads[f"W{i+1}"] = np.outer(delta, x)
            grads[f"b{i+1}"] = delta
            delta = np.dot(delta, self.weights[i].T) * self.activation.grad(x[:-1])
            x = x[:-1]

        return loss, grads

    def save(self, path: str) -> None:
        """保存模型参数到文件"""
        np.savez(path, weights=self.weights, biases=self.biases)

    def load(self, path: str) -> None:
        """从文件加载模型参数"""
        params = np.load(path)
        self.weights = params["weights"]
        self.biases = params["biases"]

    def train(self, epoch: int, dataset: Tuple[np.ndarray, np.ndarray]) -> None:
        """模型训练"""
        X, y = dataset
        for i in range(epoch):
            loss, grads = self.backprop(X, y)
            self.optimize(grads)

    def optimize(self, grads: Dict[str, np.ndarray]) -> None:
        """使用SGD更新权重"""
        for k, v in grads.items():
            if "W" in k:
                self.weights[int(k[1:])] -= self.lr * v
            if "b" in k:
                self.biases[int(k[1:])] -= self.lr * np.sum(v)

    def test(self, dataset: Tuple[np.ndarray, np.ndarray]) -> float:
        """测试模型,返回准确率"""
        X, y = dataset
        pred = self.inference(X)
        acc = np.mean(pred == y)
        return acc