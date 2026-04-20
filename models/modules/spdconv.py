"""
模块: spdconv.py
描述: 包含空间深度转换卷积 (SPDConv) 的实现，用于改进 YOLO 处理小目标的下采样过程。
"""

import torch
import torch.nn as nn
from .common import Conv


class SPD(nn.Module):
    """
    空间到深度 (Spatial-to-Depth) 转换模块。

    数学原理：
    该模块将输入张量在空间维度(H, W)上以因子 2 进行下采样，但将其无损地堆叠到通道维度(C)。
    设输入维度为 (B, C, H, W)，输出维度为 (B, 4C, H/2, W/2)。
    这避免了传统 MaxPool 或 Strided Conv 带来的细粒度信息直接丢失，非常适合小目标缺陷检测。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)。

        Returns:
            torch.Tensor: 转换后的特征图，形状为 (B, 4C, H/2, W/2)。
        """
        # 利用切片技术获取 4 个子特征图，并在通道维度进行拼接
        return torch.cat(
            [
                x[..., ::2, ::2],  # top-left
                x[..., 1::2, ::2],  # bottom-left
                x[..., ::2, 1::2],  # top-right
                x[..., 1::2, 1::2],  # bottom-right
            ],
            dim=1,
        )


class SPDConv(nn.Module):
    """
    基于 SPD 的卷积模块。
    用于替代 YOLO 主干网络中传统的步长为 2 的卷积层（Downsampling）。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
    ):
        """
        初始化 SPDConv 模块。

        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小，默认 3。
            s (int): 卷积步长，默认 1 (由于 SPD 已执行下采样，这里通常为 1)。
            p (int): 填充大小。
            g (int): 分组卷积参数。
            d (int): 空洞卷积参数。
            act (bool): 是否使用激活函数。
        """
        super().__init__()
        self.spd = SPD()
        print("SPDConv Initialized!")
        # 注意：SPD 转换后通道数变为 c1 * 4，因此后续卷积的输入通道需对应调整
        self.conv = Conv(c1 * 4, c2, k, s, p, g, d, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 经过 SPD 和 Conv 后的张量。
        """
        return self.conv(self.spd(x))

