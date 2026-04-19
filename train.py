import os
import sys
import argparse
from ultralytics import YOLO


def main(opt):
    """
    基础训练闭环管线。
    """
    print("🚀 开始 YOLO 缺陷检测模型训练...")

    # 加载模型
    # 如果是 .yaml，则初始化结构；如果是 .pt，则加载预训练权重
    model = YOLO(opt.cfg)

    # 执行训练
    # 使用 os.path.abspath 确保 project 路径是绝对的，防止嵌套目录问题
    results = model.train(
        data=opt.data,
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=opt.device,
        workers=opt.workers,
        project=os.path.abspath(opt.project),
        name=opt.name,
        # --- 小目标缺陷检测的关键超参数调优 ---
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
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
        "--cfg", type=str, default="models/yolo_improved.yaml", help="模型配置文件路径或预训练模型名(如 yolo11n.pt)"
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
