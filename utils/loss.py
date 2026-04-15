"""
模块: loss.py
描述: 包含 WIoU-v3 等针对目标检测优化的损失函数。
"""

import torch
import torch.nn as nn
import math


def bbox_iou(box1, box2, eps=1e-7):
    """
    计算两个边界框的 IoU。
    假设 box1 和 box2 的形状为 (N, 4)，格式为 [x1, y1, x2, y2] (即左上角和右下角坐标)。
    """
    # 获取坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 交集区域的坐标
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    # 交集面积
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    # 各自的面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # 联合面积
    union_area = b1_area + b2_area - inter_area + eps

    return inter_area / union_area


class WIoU_Loss(nn.Module):
    """
    Wise-IoU (WIoU) v3 损失函数计算模块。

    采用动态非单调聚焦机制，降低低质量（离群）样本的梯度惩罚，从而抵抗噪声标注的干扰。
    特别适用于小目标或者容易出现边缘模糊、标注偏差的工业缺陷检测场景。
    """

    def __init__(self, alpha: float = 1.9, delta: float = 3.0):
        """
        初始化 WIoU_Loss。

        Args:
            alpha (float): 调整非单调聚焦曲线峰值位置的超参数。
                           值越小，峰值越靠左，越早放弃低质量样本。
            delta (float): 调整非单调聚焦曲线陡峭程度的超参数。
                           值越大，曲线越陡，对低质量样本的抑制越强烈。
        """
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        # 用于记录动量以计算离群度 (outlier degree) 的移动平均值
        self.iou_mean = 1.0

    def forward(
        self, pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 WIoU v3 损失。

        Args:
            pred_bboxes (torch.Tensor): 预测边界框，形状 (N, 4)。为了兼容通用计算，假设已转换为 [x1, y1, x2, y2] 格式。
            target_bboxes (torch.Tensor): 真实边界框，形状 (N, 4)，格式 [x1, y1, x2, y2]。

        Returns:
            torch.Tensor: 返回每个边界框的 WIoU 损失，形状为 (N,)。
        """
        # 1. 计算常规的 IoU
        iou = bbox_iou(pred_bboxes, target_bboxes)

        # 2. 计算距离惩罚项 R_wiou
        # 计算两框中心点坐标
        cx_pred = (pred_bboxes[:, 0] + pred_bboxes[:, 2]) / 2
        cy_pred = (pred_bboxes[:, 1] + pred_bboxes[:, 3]) / 2
        cx_gt = (target_bboxes[:, 0] + target_bboxes[:, 2]) / 2
        cy_gt = (target_bboxes[:, 1] + target_bboxes[:, 3]) / 2

        # 计算闭包区域 (Minimum enclosing box) 的长宽 W_g, H_g
        enc_x1 = torch.min(pred_bboxes[:, 0], target_bboxes[:, 0])
        enc_y1 = torch.min(pred_bboxes[:, 1], target_bboxes[:, 1])
        enc_x2 = torch.max(pred_bboxes[:, 2], target_bboxes[:, 2])
        enc_y2 = torch.max(pred_bboxes[:, 3], target_bboxes[:, 3])

        wg = enc_x2 - enc_x1
        hg = enc_y2 - enc_y1

        # 中心点距离的平方
        dist_sq = (cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2
        # 闭包对角线距离的平方
        diag_sq = wg**2 + hg**2 + 1e-6

        # R_wiou 距离惩罚项
        R_wiou = torch.exp(dist_sq / diag_sq)

        # 3. 计算离群度 Beta (Outlier degree)
        L_iou = 1 - iou
        with torch.no_grad():
            # 动量更新 iou_mean
            self.iou_mean = 0.99 * self.iou_mean + 0.01 * L_iou.mean().item()
            # 离群度：当前样本质量与整体质量均值的比率
            beta = L_iou / (self.iou_mean + 1e-6)

        # 4. 计算非单调聚焦因子 r (Non-monotonic focusing coefficient)
        # 数学公式: r = beta / (delta * alpha^(beta - delta))
        # 当 beta 达到特定阈值（低质量），r 会减小，从而降低其梯度权重
        r = beta / (self.delta * torch.pow(self.alpha, beta - self.delta))

        # 5. 组合最终损失
        # WIoU v3 = r * R_wiou * L_iou
        wiou_loss = r * R_wiou * L_iou

        return wiou_loss
