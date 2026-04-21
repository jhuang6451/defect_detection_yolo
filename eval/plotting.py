import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# 设置绘图风格 (学术论文适用)
try:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"
except:
    pass


def discover_experiments(base_dir):
    """
    自动发现包含 results.csv 的实验目录。
    """
    exp_dirs = {}
    search_pattern = os.path.join(base_dir, "**", "results.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    for csv_path in csv_files:
        dir_path = os.path.dirname(csv_path)
        exp_name = os.path.basename(dir_path)
        if exp_name == "weights": continue
        exp_dirs[exp_name] = dir_path
        
    return dict(sorted(exp_dirs.items()))


def plot_comparison(exp_mapping, output_dir):
    """
    绘制并保存多个实验的指标对比图。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    for exp_label, dir_path in exp_mapping.items():
        csv_path = os.path.join(dir_path, "results.csv")
        if not os.path.exists(csv_path): continue

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df['Experiment'] = exp_label
        all_data.append(df)

    if not all_data:
        print("❌ 未找到任何有效的 results.csv 数据，停止绘图。")
        return

    # 1. mAP@0.5 演进曲线
    plt.figure(figsize=(10, 6))
    for df in all_data:
        label = df['Experiment'].iloc[0]
        plt.plot(df["epoch"], df["metrics/mAP50(B)"], label=label, linewidth=2)
    
    plt.xlabel("Epochs", fontweight="bold")
    plt.ylabel("mAP@0.5", fontweight="bold")
    plt.title("mAP@0.5 Evolution Comparison", fontweight="bold", pad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    map_save = os.path.join(output_dir, "map_comparison_curve.png")
    plt.savefig(map_save, dpi=300, bbox_inches="tight")
    print(f"✅ mAP 对比图已保存至: {map_save}")
    plt.close()

    # 2. Box Loss 演进曲线
    plt.figure(figsize=(10, 6))
    for df in all_data:
        label = df['Experiment'].iloc[0]
        if "val/box_loss" in df.columns:
            plt.plot(df["epoch"], df["val/box_loss"], label=label, linewidth=2)
    
    plt.xlabel("Epochs", fontweight="bold")
    plt.ylabel("Validation Box Loss", fontweight="bold")
    plt.title("Box Loss Convergence Comparison", fontweight="bold", pad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    loss_save = os.path.join(output_dir, "loss_comparison_curve.png")
    plt.savefig(loss_save, dpi=300, bbox_inches="tight")
    print(f"✅ Loss 对比图已保存至: {loss_save}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动实验图表生成工具")
    parser.add_argument("--dir", type=str, default="runs/train", help="搜索根目录")
    parser.add_argument("--output", type=str, default="eval/output", help="图表保存目录")
    opt = parser.parse_args()

    found_experiments = discover_experiments(opt.dir)
    print(f"✨ 自动发现 {len(found_experiments)} 个实验目录，准备绘图...")
    
    plot_comparison(found_experiments, opt.output)
