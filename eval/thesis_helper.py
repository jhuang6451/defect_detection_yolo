import os
import shutil
import glob
import argparse

def collect_thesis_materials(base_dir, output_dir):
    """
    自动提取并重命名各个实验中最具论文价值的图表。
    """
    # 定义需要提取的资产类型 (文件名: 论文中的别名)
    target_assets = {
        "confusion_matrix_normalized.png": "ConfusionMatrix",
        "BoxPR_curve.png": "PRCurve",
        "BoxF1_curve.png": "F1Curve",
        "val_batch0_pred.jpg": "DetectionSample_0",
        "val_batch1_pred.jpg": "DetectionSample_1",
        "results.png": "TrainingMetrics"
    }

    assets_path = os.path.join(output_dir, "thesis_assets")
    os.makedirs(assets_path, exist_ok=True)

    print(f"📂 正在从 {base_dir} 提取论文素材...")
    
    # 扫描所有实验目录
    exp_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    
    count = 0
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        # 提取关键文件
        for filename, alias in target_assets.items():
            src_file = os.path.join(exp_dir, filename)
            if os.path.exists(src_file):
                # 构造新文件名: 实验名_别名.后缀
                ext = os.path.splitext(filename)[1]
                new_name = f"{exp_name}_{alias}{ext}"
                dst_file = os.path.join(assets_path, new_name)
                
                shutil.copy2(src_file, dst_file)
                count += 1
    
    print(f"\n✅ 素材收割完成！共提取 {count} 个资产。")
    print(f"📍 请直接前往查看: {assets_path}")
    print("-" * 50)
    print("💡 论文排版建议：")
    print("1. 对比不同实验的 'ConfusionMatrix'，证明模型对小目标的召回率提升。")
    print("2. 使用 'PRCurve' 展示完全体模型在所有类别下的鲁棒性。")
    print("3. 将 'DetectionSample' 放入定性分析章节，作为 '看得见' 的改进证据。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="毕业论文素材自动化提取工具")
    parser.add_argument("--dir", type=str, default="runs/train", help="实验结果根目录")
    parser.add_argument("--out", type=str, default="eval/output", help="素材存放目录")
    opt = parser.parse_args()

    collect_thesis_materials(opt.dir, opt.out)
