"""
模块: fpn_pafpn.py
描述: 特征金字塔 (FPN) 及其双向聚合增强结构 (BiFPN) 相关模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN_Add(nn.Module):
    """
    基于加法 (Addition) 的双向特征融合模块。
    通过可学习的权重参数将不同层级的特征相加融合，适用于轻量级网络。
    """

    def __init__(self, in_channels: list, epsilon: float = 1e-4):
        """
        初始化。

        Args:
            in_channels (list): 输入的多尺度特征图的通道数列表。注意：所有输入的通道数应在进入前统一。
            epsilon (float): 防止分母为 0 的极小值。
        """
        super().__init__()
        self.epsilon = epsilon
        # 可学习的权重参数，初始化为 1
        self.w = nn.Parameter(
            torch.ones(len(in_channels), dtype=torch.float32), requires_grad=True
        )

    def forward(self, x: list) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (list of torch.Tensor): 包含多个需要融合的张量的列表。形状要求除了通道数外，H 和 W 必须一致。

        Returns:
            torch.Tensor: 融合后的特征图。
        """
        w = F.relu(self.w)
        weight_sum = torch.sum(w) + self.epsilon
        # 归一化权重
        out = sum([w[i] * x[i] for i in range(len(x))]) / weight_sum
        return out


class BiFPN_Concat(nn.Module):
    """
    基于拼接 (Concatenation) 的双向特征融合模块。
    结合了 YOLO 的 concat 风格和 BiFPN 的可学习权重思想。
    将特征乘以权重后，在通道维度进行拼接。
    """

    def __init__(self, dimension: int = 1, num_inputs: int = 2, epsilon: float = 1e-4):
        """
        初始化。

        Args:
            dimension (int): 拼接的维度，默认为 1 (通道维度)。
            num_inputs (int): 输入特征图的数量。
            epsilon (float): 防止分母为 0 的极小值。
        """
        super().__init__()
        self.d = dimension
        self.epsilon = epsilon
        self.w = nn.Parameter(
            torch.ones(num_inputs, dtype=torch.float32), requires_grad=True
        )

    def forward(self, x: list) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (list of torch.Tensor): 输入张量列表。

        Returns:
            torch.Tensor: 拼接后的张量。
        """
        w = F.relu(self.w)
        weight_sum = torch.sum(w) + self.epsilon
        # 归一化权重并与对应张量相乘，最后拼接
        weighted_x = [w[i] * x[i] / weight_sum for i in range(len(x))]
        return torch.cat(weighted_x, self.d)
