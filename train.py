import os
import sys
import argparse
from ultralytics import YOLO
import torch

# 导入 models 包并将其注册 to Ultralytics 的内部命名空间中
import ultralytics.nn.tasks as tasks
import models

# 注册基础自定义模块
tasks.GhostConv = models.SPDConv
tasks.WIoU_Loss = models.WIoU_Loss

# --- WIoU-v3 动态集成逻辑 ---
import ultralytics.utils.loss as loss_module

class WIoU_v3_BboxLoss(loss_module.BboxLoss):
    """
    自定义 BboxLoss，集成 WIoU-v3 逻辑。
    继承自官方 BboxLoss 以保持 DFL 等逻辑兼容。
    """
    def __init__(self, reg_max=16, alpha=1.9, delta=3.0):
        super().__init__(reg_max)
        self.alpha = alpha
        self.delta = delta
        self.iou_mean = 1.0
        self.momentum = 0.9  # 动量平滑

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # 1. 计算基础 IoU
        iou = loss_module.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        
        # 2. 计算 WIoU 距离惩罚 R_WIoU
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes[fg_mask].chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes[fg_mask].chunk(4, -1)
        
        px, py = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        tx, ty = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
        
        cw_x1 = torch.min(b1_x1, b2_x1)
        cw_y1 = torch.min(b1_y1, b2_y1)
        cw_x2 = torch.max(b1_x2, b2_x2)
        cw_y2 = torch.max(b1_y2, b2_y2)
        
        c2 = (cw_x2 - cw_x1)**2 + (cw_y2 - cw_y1)**2 + 1e-7
        rho2 = (px - tx)**2 + (py - ty)**2
        R_WIoU = torch.exp(rho2 / c2.detach())
        
        # 3. 动态非单调聚焦系数 r
        with torch.no_grad():
            curr_iou = iou.detach()
            self.iou_mean = self.momentum * self.iou_mean + (1 - self.momentum) * curr_iou.mean().item()
            beta = curr_iou / (self.iou_mean + 1e-7)
            r = beta / (self.delta * torch.pow(self.alpha, beta - self.delta))
        
        # 4. 组合损失
        loss_iou = (r * R_WIoU * (1.0 - iou) * weight).sum() / target_scores_sum

        # 5. DFL 损失
        if self.dfl_loss:
            target_ltrb = loss_module.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


def main(opt):
    """
    基础训练闭环管线。
    """
    print(f"🚀 开始 YOLO 缺陷检测模型训练... (配置: {opt.cfg})")

    # --- 动态补丁: 架构保护 ---
    cfg_name_lower = os.path.basename(opt.cfg).lower()
    if "bifpn" in cfg_name_lower:
        print("🛠️ 探测到 BiFPN 配置，已将底层 Concat 动态替换为 BiFPN_Concat。")
        tasks.Concat = models.BiFPN_Concat
    else:
        import ultralytics.nn.modules.conv as conv
        tasks.Concat = conv.Concat

    # --- 动态补丁: 损失函数切换 ---
    if opt.wiou:
        print(f"⚖️ 开启 WIoU-v3 损失函数 (alpha={opt.wiou_alpha}, delta={opt.wiou_delta})。")
        loss_module.BboxLoss = lambda reg_max: WIoU_v3_BboxLoss(reg_max, alpha=opt.wiou_alpha, delta=opt.wiou_delta)
    else:
        from ultralytics.utils.loss import BboxLoss as StdBoxLoss
        loss_module.BboxLoss = StdBoxLoss

    # 加载模型
    model = YOLO(opt.cfg)

    # 执行训练
    results = model.train(
        data=opt.data,
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=opt.device,
        workers=opt.workers,
        project=os.path.abspath(opt.project),
        name=opt.name,
        # --- 使用从命令行/网格搜索传入的参数 ---
        lr0=opt.lr0,
        lrf=opt.lrf,
        warmup_epochs=opt.warmup_epochs,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        iou=0.4,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        patience=50,
        save=True,
        val=True,
    )

    print(f"✅ 训练结束！模型及日志保存在: {os.path.join(opt.project, opt.name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 缺陷检测模型训练脚本")
    parser.add_argument(
        "--cfg", type=str, default="models/yolo_improved.yaml", help="模型配置文件路径或预训练模型名"
    )
    parser.add_argument(
        "--data", type=str, default="datasets/data.yaml", help="数据集配置文件路径"
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--project", default="runs/train", help="保存项目的根目录")
    parser.add_argument("--name", default="exp_improved", help="保存本次实验的目录名")
    
    # 核心超参数 (用于 Grid Search)
    parser.add_argument("--lr0", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率因子 (lr0 * lrf)")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="预热轮次")
    
    # WIoU 相关参数
    parser.add_argument("--wiou", action="store_true", help="是否使用 WIoU-v3 损失函数")
    parser.add_argument("--wiou_alpha", type=float, default=1.9, help="WIoU alpha 参数")
    parser.add_argument("--wiou_delta", type=float, default=3.0, help="WIoU delta 参数")

    opt = parser.parse_args()

    # 为了防止路径问题，做一些简单的检查
    if not os.path.exists(opt.cfg) and not opt.cfg.endswith('.pt'):
       print(f"⚠️ 警告: 模型配置文件 {opt.cfg} 不存在。")
    if not os.path.exists(opt.data):
       print(f"⚠️ 警告: 数据集配置文件 {opt.data} 不存在。")

    main(opt)
