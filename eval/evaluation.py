import os
import argparse
import pandas as pd
from ultralytics import YOLO
import sys
import glob

# 添加父目录到路径以便导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.metrics import compute_flops
except ImportError:
    def compute_flops(model, input_size): return "N/A", "N/A"


def discover_experiments(base_dir):
    """
    自动发现包含 results.csv 的实验目录。
    """
    exp_dirs = {}
    # 扫描 runs/train 及其子目录
    search_pattern = os.path.join(base_dir, "**", "results.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    for csv_path in csv_files:
        dir_path = os.path.dirname(csv_path)
        exp_name = os.path.basename(dir_path)
        # 排除 weights 目录本身（虽然通常不会有 csv）
        if exp_name == "weights":
            continue
        exp_dirs[exp_name] = dir_path
        
    # 按文件夹名称排序，确保 01, 03, 04 有序
    sorted_exps = dict(sorted(exp_dirs.items()))
    return sorted_exps


def evaluate_models(exp_mapping, img_size, output_dir):
    """
    提取关键指标并生成对比表格。
    """
    results = []
    os.makedirs(output_dir, exist_ok=True)

    if not exp_mapping:
        print("❌ 未发现任何有效的训练结果目录。")
        return

    for exp_label, dir_path in exp_mapping.items():
        csv_path = os.path.join(dir_path, "results.csv")
        weights_path = os.path.join(dir_path, "weights", "best.pt")
        
        print(f"🔍 正在处理: {exp_label}")
        
        # 1. 提取 CSV 指标
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # 找到 mAP 最高的那个 Epoch
        best_map50 = df["metrics/mAP50(B)"].max()
        best_row = df.loc[df["metrics/mAP50(B)"].idxmax()]
        map50_95 = best_row["metrics/mAP50-95(B)"]
        
        # 2. 计算模型复杂度
        flops, params = "N/A", "N/A"
        if os.path.exists(weights_path):
            try:
                # 暂时禁用打印以保持输出整洁
                sys.stdout = open(os.devnull, 'w')
                model = YOLO(weights_path)
                flops, params = compute_flops(model.model, (1, 3, img_size, img_size))
                sys.stdout = sys.__stdout__
            except:
                sys.stdout = sys.__stdout__

        results.append({
            "Experiment": exp_label,
            "mAP@0.5": f"{best_map50:.4f}",
            "mAP@0.5:0.95": f"{map50_95:.4f}",
            "Params (M)": params,
            "FLOPs (G)": flops,
        })

    # 生成总结表格
    final_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("🏆 实验指标自动汇总表 (Automated Ablation Summary)")
    print("=" * 70)
    
    try:
        from tabulate import tabulate
        print(tabulate(final_df, headers='keys', tablefmt='github', showindex=False))
    except ImportError:
        print(final_df.to_string(index=False))

    # 保存
    csv_out = os.path.join(output_dir, "ablation_summary.csv")
    final_df.to_csv(csv_out, index=False)
    print(f"\n✅ 对比报表已保存至: {csv_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动实验指标提取工具")
    parser.add_argument("--dir", type=str, default="runs/train", help="搜索根目录")
    parser.add_argument("--imgsz", type=int, default=640, help="计算 FLOPs 的尺寸")
    parser.add_argument("--output", type=str, default="eval/output", help="结果保存目录")
    opt = parser.parse_args()

    # 1. 自动发现
    found_experiments = discover_experiments(opt.dir)
    print(f"✨ 自动发现 {len(found_experiments)} 个实验目录。")
    
    # 2. 评估
    evaluate_models(found_experiments, opt.imgsz, opt.output)
