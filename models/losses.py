import torch
import torch.nn as nn

class WIoU_Loss(nn.Module):
    """
    Wise-IoU (WIoU) v3 损失函数
    
    此损失函数引入了动态非单调聚焦机制（Dynamic Non-monotonic Focusing Mechanism）。
    针对工业零部件数据集中可能存在的“低质量样本”（如模糊缺陷、噪声标注），WIoU-v3 根据预测框
    的“离群度（Outlier Degree）”动态分配梯度。它优先优化那些普通的高质量样本，同时抑制
    低质量样本对网络梯度的负面影响，从而加速收敛并提高整体的模型泛化性能。
    
    参数:
        alpha (float, optional): 用于控制非单调聚焦曲线峰值位置的超参数。默认值为 1.9。
        delta (float, optional): 用于控制离群度降权斜率力度的超参数。默认值为 3.0。
    """
    def __init__(self, alpha=1.9, delta=3.0):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.momentum = 0.95
        # 初始化平均 IoU，使用 EMA（指数移动平均）来平滑更新
        self.iou_mean = 1.0 
        
    def forward(self, pred_bboxes, target_bboxes):
        """
        前向传播函数。
        
        参数:
            pred_bboxes (torch.Tensor): 预测的边界框，形状为 (N, 4)，格式为 [x_c, y_c, w, h]。
            target_bboxes (torch.Tensor): 目标的边界框，形状为 (N, 4)，格式为 [x_c, y_c, w, h]。
            
        返回:
            torch.Tensor: 计算出的 WIoU-v3 损失值组成的张量。
        """
        # --- 1. 计算 IoU ---
        # 提取中心点和宽高
        px, py, pw, ph = pred_bboxes[:, 0], pred_bboxes[:, 1], pred_bboxes[:, 2], pred_bboxes[:, 3]
        tx, ty, tw, th = target_bboxes[:, 0], target_bboxes[:, 1], target_bboxes[:, 2], target_bboxes[:, 3]

        # 转换为左上角和右下角坐标 [x1, y1, x2, y2]
        px1, py1, px2, py2 = px - pw/2, py - ph/2, px + pw/2, py + ph/2
        tx1, ty1, tx2, ty2 = tx - tw/2, ty - th/2, tx + tw/2, ty + th/2

        # 计算交集区域
        inter_x1 = torch.max(px1, tx1)
        inter_y1 = torch.max(py1, ty1)
        inter_x2 = torch.min(px2, tx2)
        inter_y2 = torch.min(py2, ty2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # 计算各自的面积以及并集区域
        pred_area = pw * ph
        target_area = tw * th
        union_area = pred_area + target_area - inter_area
        
        # 加上微小常数防止除 0
        iou = inter_area / (union_area + 1e-7)

        # --- 2. 计算距离惩罚项 R_WIoU ---
        # 最小外接框的坐标
        cw_x1 = torch.min(px1, tx1)
        cw_y1 = torch.min(py1, ty1)
        cw_x2 = torch.max(px2, tx2)
        cw_y2 = torch.max(py2, ty2)

        # 最小外接矩形对角线的平方加上常数
        c2 = (cw_x2 - cw_x1)**2 + (cw_y2 - cw_y1)**2 + 1e-7
        # 中心点距离的平方
        rho2 = (px - tx)**2 + (py - ty)**2

        # 计算 WIoU-v1 基础距离惩罚项的指数部分
        exp_factor = torch.exp(rho2 / c2)
        
        # WIoU-v1 损失计算
        L_WIoU_v1 = exp_factor * (1 - iou)

        # --- 3. WIoU-v3 动态非单调聚焦机制 ---
        with torch.no_grad():
            current_mean_iou = iou.mean()
            # 动态更新 iou_mean
            self.iou_mean = self.momentum * self.iou_mean + (1 - self.momentum) * current_mean_iou.item()
            
            # 计算离群度 beta
            # 如果 iou 很高，则 beta 很大；如果 iou 很低，则 beta 很小。
            beta = iou.detach() / (self.iou_mean + 1e-7)
            
            # 这里的非单调系数计算：让极差的样本（低 beta）获得的权重极小，保护模型不被脏数据拉偏；
            # 同时适当降低极其容易样本的权重，主攻“一般困难”的有效样本。
            non_monotonic_factor = beta / (self.delta * torch.pow(self.alpha, self.alpha - beta))

        # --- 4. 最终损失 ---
        L_WIoU_v3 = non_monotonic_factor * L_WIoU_v1
        
        # 返回平均损失
        return L_WIoU_v3.mean()
