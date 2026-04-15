import os
import sys
import argparse
from ultralytics import YOLO


def main(opt):
    """
    基础训练闭环管线。
    利用 Ultralytics 的训练管道，但使用我们改进的 yolo_improved.yaml。
    """
    print("🚀 开始 YOLO 缺陷检测模型训练...")

    # 1. 加载模型配置 (如果有预训练权重，可在此加入)
    # 此处加载我们包含 SPDConv 等改进结构的 YAML
    model = YOLO(opt.cfg)

    # 2. 执行训练
    # Ultralytics 的 train 接受大量字典参数。对于小目标检测，
    # 参考 docs/hyperparameter_tuning_guide.md 设定一些关键超参数。
    results = model.train(
        data=opt.data,
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=opt.device,
        workers=opt.workers,
        project=opt.project,
        name=opt.name,
        # --- 小目标缺陷检测的关键超参数调优 ---
        lr0=0.001,  # 较低的初始学习率
        lrf=0.01,  # 最终学习率乘数
        warmup_epochs=3.0,  # 预热 epochs，供 WIoU-v3 稳定动量
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # 匹配策略
        iou=0.4,  # 测试/验证时的 NMS 阈值 (可根据情况调低)
        # 注意: box iou_t 在 ultralytics 中较难直接通过命令行传参，
        # 实际可能需要修改 default.yaml 或传入特定的 hyp.yaml。
        # 数据增强 (针对小目标)
        mosaic=1.0,  # 开启 Mosaic
        mixup=0.0,  # 关闭 Mixup 以免混淆背景
        copy_paste=0.1,  # 可选的 copy-paste
        # 其他
        patience=50,  # Early stopping patience
        save=True,  # 保存 best.pt 和 last.pt
        val=True,  # 训练中途进行验证
    )

    print(f"✅ 训练结束！模型及日志保存在: {os.path.join(opt.project, opt.name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 缺陷检测模型训练脚本")
    parser.add_argument(
        "--cfg", type=str, default="models/yolo_improved.yaml", help="模型配置文件路径"
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

    opt = parser.parse_args()

    # 为了防止路径问题，做一些简单的检查
    if not os.path.exists(opt.cfg):
        print(f"⚠️ 警告: 模型配置文件 {opt.cfg} 不存在。")
    if not os.path.exists(opt.data):
        print(f"⚠️ 警告: 数据集配置文件 {opt.data} 不存在。")

    main(opt)
