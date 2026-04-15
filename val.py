import os
import argparse
from ultralytics import YOLO


def main(opt):
    """
    模型验证闭环管线。
    用于在测试集/验证集上评估训练好的模型，输出 mAP 等指标。
    """
    print(f"🚀 开始模型验证，加载权重: {opt.weights}")

    if not os.path.exists(opt.weights):
        print(f"❌ 错误: 权重文件 {opt.weights} 不存在。请先运行 train.py 进行训练。")
        return

    model = YOLO(opt.weights)

    # 执行验证
    metrics = model.val(
        data=opt.data,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=opt.device,
        conf=opt.conf,  # 预测的置信度阈值
        iou=opt.iou_nms,  # NMS 的 IoU 阈值
        project=opt.project,
        name=opt.name,
        save_json=True,  # 保存 COCO 格式的 json 用于后续独立评估
    )

    print("\n✅ 验证结束！核心指标如下:")
    print(f"  mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")

    # 可以在此处附加独立的 FLOPs 等计算
    # from utils.metrics import compute_flops
    # flops, params = compute_flops(model.model, (1, 3, opt.imgsz, opt.imgsz))
    # print(f"  模型计算量 (FLOPs): {flops}")
    # print(f"  模型参数量 (Params): {params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 缺陷检测模型验证脚本")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/exp_improved/weights/best.pt",
        help="训练好的模型权重路径",
    )
    parser.add_argument(
        "--data", type=str, default="datasets/data.yaml", help="数据集配置文件路径"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="预测置信度阈值")
    parser.add_argument("--iou_nms", type=float, default=0.45, help="NMS 的 IoU 阈值")
    parser.add_argument("--project", default="runs/val", help="保存项目的根目录")
    parser.add_argument("--name", default="exp_val", help="保存本次验证的目录名")

    opt = parser.parse_args()
    main(opt)
