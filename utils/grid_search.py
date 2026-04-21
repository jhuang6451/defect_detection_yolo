import subprocess
import itertools
import os

# ================= 配置区域 =================
# 减少搜索空间以节省时间
alphas = [1.6, 1.9, 2.2]
deltas = [2.5, 3.0]
lrs = [0.001, 0.0005]  # 配合 WIoU，建议使用稍小的学习率

CFG = "models/yolo26n_spdconv_bifpn.yaml"
DATA = "datasets/GC10-DET-YOLO/data.yaml"
EPOCHS = 40  # 40 轮足以看出收敛趋势
BATCH = 16
# ==========================================

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
        "--project", "runs/grid_search"
    ]
    
    # 运行并等待结束
    subprocess.run(cmd, check=True)

def main():
    # 创建保存目录
    os.makedirs("runs/grid_search", exist_ok=True)
    
    # 生成所有组合
    combinations = list(itertools.product(alphas, deltas, lrs))
    total = len(combinations)
    
    print(f"🎯 准备进行网格搜索，共 {total} 组试验。")
    
    for i, (a, d, lr) in enumerate(combinations):
        print(f"\n进度: {i+1}/{total}")
        try:
            run_experiment(a, d, lr)
        except Exception as e:
            print(f"❌ 实验 {i+1} 失败: {e}")
            continue

    print("\n✅ 所有网格搜索试验已完成！结果保存在 runs/grid_search 目录下。")
    print("建议下一步：运行分析脚本对比各组 mAP，选出最强参数。")

if __name__ == "__main__":
    main()
