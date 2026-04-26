import subprocess
import itertools
import os

# ================= 配置区域 (09 实验：yolo26s + 增强数据集的 WIoU 参数搜索) =================
# 重点探索更高容忍度的 alpha，验证正则化冲突猜想
alphas = [1.6, 1.9, 2.2, 2.5]
deltas = [2.5, 3.0]
lrs = [0.001]  # 固定学习率，控制搜索空间

CFG = "models/yolo26s_spdconv_bifpn.yaml"
DATA = "datasets/GC10-DET-YOLO-AUGMENTED/data.yaml"
EPOCHS = 40  # 40 轮足以看出收敛趋势
BATCH = 16
PROJECT_DIR = "runs/grid_search_09"
# =======================================================================================

def run_experiment(a, d, lr):
    exp_name = f"grid_a{a}_d{d}_lr{lr}"
    print(f"\n🚀 开始实验: {exp_name} (Alpha={a}, Delta={d}, LR={lr})")
    
    cmd = [
        "uv", "run", "train.py",
        "--cfg", CFG,
        "--data", DATA,
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH),
        "--name", exp_name,
        "--lr0", str(lr),
        "--wiou",
        "--wiou_alpha", str(a),
        "--wiou_delta", str(d),
        "--project", PROJECT_DIR
    ]
    
    # 运行并等待结束
    subprocess.run(cmd, check=True)

def main():
    # 创建保存目录
    os.makedirs(PROJECT_DIR, exist_ok=True)
    
    # 生成所有组合
    combinations = list(itertools.product(alphas, deltas, lrs))
    total = len(combinations)
    
    print(f"🎯 准备进行网格搜索，共 {total} 组试验。")
    print(f"配置文件: {CFG}")
    print(f"数据集: {DATA}")
    
    for i, (a, d, lr) in enumerate(combinations):
        print(f"\n进度: {i+1}/{total}")
        try:
            run_experiment(a, d, lr)
        except Exception as e:
            print(f"❌ 实验 {i+1} 失败: {e}")
            continue

    print(f"\n✅ 所有网格搜索试验已完成！结果保存在 {PROJECT_DIR} 目录下。")

if __name__ == "__main__":
    main()
