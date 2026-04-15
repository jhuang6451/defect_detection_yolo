import os
import argparse
import pandas as pd
from ultralytics import YOLO
import sys

# 添加父目录到路径以便导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import compute_flops


def evaluate_models(exp_dirs, img_size):
    """
    遍历不同的实验结果目录，提取关键指标并生成对比表格。
    用于论文中的消融实验 (Ablation Study)。
    """
    results = []

    for exp_name, dir_path in exp_dirs.items():
        weights_path = os.path.join(dir_path, "weights", "best.pt")
        if not os.path.exists(weights_path):
            print(f"⚠️ 警告: 实验 {exp_name} 的权重文件 {weights_path} 不存在，跳过。")
            continue

        print(f"正在评估: {exp_name}")
        model = YOLO(weights_path)

        # 提取验证结果
        # 注意：此处为演示，实际可能需要读取 results.csv
        # 这里为了简化，直接调用模型，但这可能比较慢。
        # 推荐的做法是读取 Ultralytics 在 runs/train/exp 目录下生成的 results.csv
        csv_path = os.path.join(dir_path, "results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 获取最后一行或 best epoch 的数据
            df.columns = df.columns.str.strip()  # 去除列名空格
            best_mAP50 = df["metrics/mAP50(B)"].max()
            best_mAP50_95 = df["metrics/mAP50-95(B)"].max()
        else:
            best_mAP50 = "N/A"
            best_mAP50_95 = "N/A"

        # 计算参数量和 FLOPs
        flops, params = compute_flops(model.model, (1, 3, img_size, img_size))

        results.append(
            {
                "Model Configuration": exp_name,
                "mAP@0.5": best_mAP50,
                "mAP@0.5:0.95": best_mAP50_95,
                "Params": params,
                "FLOPs": flops,
            }
        )

    # 生成 DataFrame 并输出
    final_df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("消融实验对比结果 (Ablation Study)")
    print("=" * 50)
    print(final_df.to_markdown(index=False))

    # 保存到 CSV
    output_path = os.path.join("eval", "ablation_results.csv")
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ 对比结果已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="核心指标计算与对比脚本")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="输入图像尺寸用于计算 FLOPs"
    )
    opt = parser.parse_args()

    # 定义你要对比的实验目录 (需手动指定训练后的目录)
    # 例如：
    experiments = {
        "Baseline (YOLO26)": "runs/train/exp_baseline",
        "+ SPDConv": "runs/train/exp_spdconv",
        "+ WIoU-v3": "runs/train/exp_wiou",
        "Proposed (+SPDConv+WIoUv3)": "runs/train/exp_improved",
    }

    evaluate_models(experiments, opt.imgsz)
