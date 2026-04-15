import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格 (学术论文适用)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "serif"


def plot_loss_comparison(experiments, output_dir):
    """
    绘制并保存多个实验的 Loss 收敛曲线对比图。
    突出 WIoU-v3 的加速收敛效果。
    """
    plt.figure(figsize=(8, 6))

    for exp_name, dir_path in experiments.items():
        csv_path = os.path.join(dir_path, "results.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # 假设我们要画的是验证集的总 loss 或 box loss
        # 这里以 val/box_loss 为例
        if "val/box_loss" in df.columns:
            plt.plot(df["epoch"], df["val/box_loss"], label=exp_name, linewidth=2)

    plt.xlabel("Epochs", fontweight="bold")
    plt.ylabel("Validation Box Loss", fontweight="bold")
    plt.title("Convergence Comparison of Box Loss", fontweight="bold", pad=15)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "loss_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Loss 对比图已保存至: {save_path}")
    plt.close()


def plot_map_comparison(experiments, output_dir):
    """
    绘制并保存多个实验的 mAP 曲线对比图。
    """
    plt.figure(figsize=(8, 6))

    for exp_name, dir_path in experiments.items():
        csv_path = os.path.join(dir_path, "results.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        if "metrics/mAP50(B)" in df.columns:
            plt.plot(df["epoch"], df["metrics/mAP50(B)"], label=exp_name, linewidth=2)

    plt.xlabel("Epochs", fontweight="bold")
    plt.ylabel("mAP@0.5", fontweight="bold")
    plt.title("mAP@0.5 Evolution Over Epochs", fontweight="bold", pad=15)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "map_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ mAP 对比图已保存至: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学术图表绘制脚本")
    opt = parser.parse_args()

    experiments = {
        "Baseline": "runs/train/exp_baseline",
        "Proposed (+SPDConv+WIoUv3)": "runs/train/exp_improved",
    }

    os.makedirs("eval/plots", exist_ok=True)

    print("开始绘制图表...")
    plot_loss_comparison(experiments, "eval/plots")
    plot_map_comparison(experiments, "eval/plots")
