import torch
import torch.nn as nn

class SPDConv(nn.Module):
    """
    空间深度转换卷积模块 (Space-to-Depth Convolution)
    
    此模块特别为小目标缺陷检测设计。用于替换传统的步长卷积（Strided Convolution）或池化层。
    传统的下采样会导致细粒度特征快速丢失，而 SPDConv 通过将空间维度（高和宽）重塑到通道维度，
    在不丢失信息的前提下将空间尺寸缩小一半，通道数增加 4 倍。随后通过一个无步长的卷积核
    来调整最终的输出通道数，从而极大地保留了小目标特征。
    
    参数:
        inc (int): 输入特征图的通道数。
        outc (int): 输出特征图的通道数。
    """
    def __init__(self, inc, outc):
        super().__init__()
        # 经过 SPD 操作后，通道数变为原来的 4 倍
        self.conv = nn.Conv2d(inc * 4, outc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(outc)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (Batch_Size, Channels, Height, Width)
            
        返回:
            torch.Tensor: 输出张量，形状为 (Batch_Size, outc, Height/2, Width/2)
        """
        # 将输入张量在空间维度上切分为 4 个子块
        # x0: 左上角子块
        x0 = x[:, :, 0::2, 0::2]
        # x1: 左下角子块
        x1 = x[:, :, 1::2, 0::2]
        # x2: 右上角子块
        x2 = x[:, :, 0::2, 1::2]
        # x3: 右下角子块
        x3 = x[:, :, 1::2, 1::2]
        
        # 将这 4 个子块在通道维度 (dim=1) 上进行拼接
        # 拼接后的张量形状由 (B, C, H/2, W/2) 变为 (B, C*4, H/2, W/2)
        x = torch.cat([x0, x1, x2, x3], dim=1)
        
        # 通过 3x3 卷积层、BN层和激活函数，调整通道数为 outc
        return self.act(self.bn(self.conv(x)))
