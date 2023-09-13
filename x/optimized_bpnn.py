#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from typing import Dict, List, Tuple

import numpy as np

from activation import Sigmoid, Activation, Tanh, ReLU
from losses import MSELoss, Loss, CrossEntropyLoss, L1Loss


class NeuralNetwork:
    def __init__(
        self,
        layers: List[int],
        lr: float = 0.01,
        momentum: float = 0.9,
        activation: Activation = Sigmoid(),
        loss_fn: Loss = MSELoss(),
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

        self.activation = activation
        self.loss_fn = loss_fn

    def inference(self, inputs: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """前向传播,返回输出和中间变量"""
        x = inputs
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            # x * w + b
            x = np.dot(x, w) + b
            # 激活函数
            x = self.activation(x)
            activations.append(x)
        return x, activations

    def backprop(
        self, inputs: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """反向传播,返回损失和梯度"""
        # 前向传播
        pred, activations = self.inference(inputs)

        # 计算损失
        grads = {}
        loss = self.loss_fn(pred, labels)
        delta = self.loss_fn.grad(pred, labels) * self.activation.grad(pred)

        # 反向传播计算梯度
        for i in reversed(range(len(self.weights))):
            grads[f"W{i+1}"] = np.outer(delta, activations[i])
            grads[f"b{i+1}"] = delta
            delta = np.dot(delta, self.weights[i].T) * self.activation.grad(
                activations[i]
            )

        return loss, grads

    def save(self, path: str) -> None:
        """保存模型参数到文件"""

        model_dict = {
            "layers": self.layers,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "lr": self.lr,
            "momentum": self.momentum,
            "activation": str(self.activation),
            "loss_fn": str(self.loss_fn),
        }

        with open(path, "w") as f:
            json.dump(model_dict, f)

    def load(self, path: str) -> None:
        """从文件加载模型参数"""
        with open(path) as f:
            model_dict = json.load(f)

        self.layers = model_dict["layers"]
        self.weights = [np.array(w) for w in model_dict["weights"]]
        self.biases = [np.array(b) for b in model_dict["biases"]]
        self.lr = model_dict["lr"]
        self.momentum = model_dict["momentum"]

        # Note: You'll need to implement a way to correctly restore the activation and loss functions here
        match model_dict["activation"]:
            case "Sigmoid()":
                self.activation = Sigmoid()
            case "Tanh()":
                self.activation = Tanh()
            case "ReLU()":
                self.activation = ReLU()
            case _:
                raise ValueError(
                    f"Unknown activation function: {model_dict['activation']}"
                )
        match model_dict["loss_fn"]:
            case "MSELoss()":
                self.loss_fn = MSELoss()
            case "CrossEntropyLoss()":
                self.loss_fn = CrossEntropyLoss()
            case "L1Loss()":
                self.loss_fn = L1Loss()
            case _:
                raise ValueError(f"Unknown loss function: {model_dict['loss_fn']}")

    def optimize(self, grads: Dict[str, np.ndarray]) -> None:
        """使用SGD更新权重"""
        for k, v in grads.items():
            if "W" in k:
                self.weights[int(k[1:])] -= self.lr * v
            if "b" in k:
                self.biases[int(k[1:])] -= self.lr * np.sum(v)

    def train(self, epoch: int, dataset: Tuple[np.ndarray, np.ndarray]) -> None:
        """模型训练
        epoch: 训练轮数
        dataset: 训练数据集, 分别是输入和标签
        """
        for _ in range(epoch):
            _, grads = self.backprop(*dataset)
            self.optimize(grads)

    def test(self, dataset: Tuple[np.ndarray, np.ndarray]) -> None:
        """测试模型,返回准确率"""
        X, y = dataset
        pred, _ = self.inference(X)

        assert np.allclose(pred, y, atol=1e-2)
