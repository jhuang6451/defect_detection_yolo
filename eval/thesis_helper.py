import os
import shutil
import glob
import argparse

def collect_thesis_materials(base_dir, output_dir):
    """
    只提取大纲中用到的关键实验结果图表。
    """
    # 核心实验白名单 (使用重命名后的规整名称)
    whitelist = [
        "Baseline_N_Orig",
        "Baseline_S_Aug",
        "Proposed_N_SPD_Orig",
        "Proposed_N_SPD_BiFPN_Orig",
        "Proposed_N_Full_v2_Orig",
        "Proposed_S_Full_Aug",
        "Proposed_S_Arch_Aug_Confirm",
        "Proposed_S_WIoU_Aug",
        "Proposed_N_Full_v2_Aug"  # 负面论证: 容量陷阱证据
    ]

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

    print(f"📂 正在从 {base_dir} 提取核心论文素材...")
    
    count = 0
    for exp_name in whitelist:
        exp_dir = os.path.join(base_dir, exp_name)
        if not os.path.exists(exp_dir):
            print(f"⚠️ 警告: 找不到实验目录 {exp_name}，跳过。")
            continue
            
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
    
    print(f"\n✅ 核心素材收割完成！共提取 {count} 个资产。")
    print(f"📍 请直接前往查看: {assets_path}")
    print("-" * 50)
    print("💡 论文排版特别建议：")
    print("1. 重点对比 Baseline_S_Aug 与 Proposed_S_Full_Aug 的 Precision 曲线。")
    print("2. 使用 Proposed_N_Full_v2_Aug 的低迷指标来支撑‘容量陷阱’章节。")
    print("3. 在消融章节，将 11, 12, 13, 10 四组实验的 PR 曲线拼成一张大图。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="毕业论文素材自动化提取工具 (精简版)")
    parser.add_argument("--dir", type=str, default="runs/train", help="实验结果根目录")
    parser.add_argument("--out", type=str, default="eval/output", help="素材存放目录")
    opt = parser.parse_args()

    collect_thesis_materials(opt.dir, opt.out)
