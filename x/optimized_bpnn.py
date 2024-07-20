#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from typing import Dict, List, Tuple

import numpy as np
from activation import Activation, LeakyReLU, ReLU, Sigmoid, Tanh
from losses import CrossEntropyLoss, L1Loss, Loss, MSELoss


class NeuralNetwork:
    def __init__(
        self,
        layers: List[int],
        lr: float = 0.01,
        momentum: float = 0.9,
        activation: Activation = Sigmoid(),
        loss_fn: Loss = MSELoss(),
        batch_size: int = 32,
        l2_reg: float = 0.0,
    ) -> None:
        """
        构造函数
        layers: 网络层数列表 - 例如[2,4,1]表示输入层2个节点,隐层4个节点,输出层1个节点
        lr: 学习率 - 梯度下降更新权重时的步长
        momentum: 动量因子 - 用于给梯度附加动量
        activation: 激活函数 - 用于隐层的激活, 默认为Sigmoid, 可选Tanh, ReLU, LeakyReLU
        loss_fn: 损失函数 - 用于计算损失, 默认为MSELoss, 可选CrossEntropyLoss, L1Loss
        batch_size: 批大小 - 用于训练时的批量梯度下降
        l2_reg: L2正则化系数 - 用于损失函数的正则化项
        """
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.l2_reg = l2_reg

        # 初始化权重和偏置 (使用He初始化)
        self.weights = [
            np.random.randn(y, x) * np.sqrt(2 / x)
            for x, y in zip(layers[:-1], layers[1:])
        ]
        self.biases = [np.zeros(y) for y in layers[1:]]

        self.activation = activation
        self.loss_fn = loss_fn

        # 初始化动量项
        self.vw = [np.zeros_like(w) for w in self.weights]
        self.vb = [np.zeros_like(b) for b in self.biases]

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

    def optimize(self, grads: Dict[str, np.ndarray]) -> None:
        """使用带动量的SGD更新权重"""
        for k, v in grads.items():
            if "W" in k:
                i = int(k[1:])
                self.vw[i] = self.momentum * self.vw[i] + (1 - self.momentum) * v
                self.weights[i] -= self.lr * (
                    self.vw[i] + self.l2_reg * self.weights[i]
                )
            if "b" in k:
                i = int(k[1:])
                self.vb[i] = self.momentum * self.vb[i] + (1 - self.momentum) * np.sum(
                    v
                )
                self.biases[i] -= self.lr * (self.vb[i] + self.l2_reg * self.biases[i])

    def train(self, epoch: int, dataset: Tuple[np.ndarray, np.ndarray]) -> None:
        """模型训练
        epoch: 训练轮数
        dataset: 训练数据集 - [输入, 标签]
        """
        n_samples = dataset[0].shape[0]
        n_batches = n_samples // self.batch_size

        for _ in range(epoch):
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size

                batch_data = dataset[0][start_idx:end_idx]
                batch_labels = dataset[1][start_idx:end_idx]

                _, grads = self.backprop(batch_data, batch_labels)
                self.optimize(grads)

    def test(self, dataset: Tuple[np.ndarray, np.ndarray]) -> None:
        """测试模型,返回准确率"""
        x, y = dataset
        pred, _ = self.inference(x)

        assert np.allclose(pred, y, atol=1e-2)

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
            case "LeakyReLU()":
                self.activation = LeakyReLU()
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
