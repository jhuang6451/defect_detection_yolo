import os
import cv2
import glob
import random
import numpy as np
from collections import defaultdict

# ================= 配置区域 =================
DATASET_DIR = "datasets/GC10-DET-YOLO-AUGMENTED"
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "images/train")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "labels/train")

TARGET_INSTANCES = 700  # 目标样本数

# Baseline 漏检率 (Leakage Rates) -> 决定 Copy-Paste 的权重
# 格式: 类别ID: 漏检率 (0.0 ~ 1.0)
LEAKAGE_RATES = {
    0: 0.10,  # punching_hole (打孔)
    1: 0.13,  # welding_line (焊缝)
    2: 0.40,  # crescent_gap (月牙弯)
    3: 0.20,  # water_spot (水斑)
    4: 0.30,  # oil_spot (油斑)
    5: 0.05,  # silk_spot (丝斑)
    6: 0.85,  # inclusion (夹杂)
    7: 1.00,  # rolled_pit (轧坑) -> 100% 漏检！
    8: 0.83,  # crease (折痕)
    9: 0.50,  # waist_folding (腰折)
}
# ==========================================

def get_current_distribution():
    """统计当前训练集的类别分布，并建立 图像->标签 的映射"""
    distribution = defaultdict(int)
    image_to_instances = defaultdict(list)
    
    label_files = glob.glob(os.path.join(TRAIN_LBL_DIR, "*.txt"))
    for lbl_path in label_files:
        img_name = os.path.basename(lbl_path).replace('.txt', '.jpg')
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue
            
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    c, x, y, w, h = map(float, parts)
                    c = int(c)
                    distribution[c] += 1
                    image_to_instances[img_path].append((c, x, y, w, h))
    return distribution, image_to_instances

def extract_patches(image_to_instances):
    """从原图中抠出所有的缺陷图块 (Patches)，用于 Copy-Paste"""
    patches = defaultdict(list)
    for img_path, instances in image_to_instances.items():
        img = cv2.imread(img_path)
        if img is None: continue
        H, W = img.shape[:2]
        
        for inst in instances:
            c, x, y, w, h = inst
            px, py = int(x * W), int(y * H)
            pw, ph = int(w * W), int(h * H)
            
            x1, y1 = max(0, px - pw // 2), max(0, py - ph // 2)
            x2, y2 = min(W, px + pw // 2), min(H, py + ph // 2)
            
            if x2 > x1 and y2 > y1:
                patch = img[y1:y2, x1:x2].copy()
                patches[c].append(patch)
    return patches

def apply_image_processing(img, instances):
    """基础图像处理：随机翻转、亮度对比度调整"""
    new_img = img.copy()
    new_instances = []
    
    # 50% 概率水平翻转
    if random.random() > 0.5:
        new_img = cv2.flip(new_img, 1)
        for c, x, y, w, h in instances:
            new_instances.append((c, 1.0 - x, y, w, h))
    else:
        new_instances = list(instances)
        
    # 50% 概率垂直翻转
    if random.random() > 0.5:
        new_img = cv2.flip(new_img, 0)
        temp_instances = []
        for c, x, y, w, h in new_instances:
            temp_instances.append((c, x, 1.0 - y, w, h))
        new_instances = temp_instances
        
    # 随机亮度和对比度扰动
    alpha = random.uniform(0.8, 1.2)  # 对比度
    beta = random.randint(-30, 30)    # 亮度
    new_img = cv2.convertScaleAbs(new_img, alpha=alpha, beta=beta)
    
    return new_img, new_instances

def apply_copy_paste(bg_img, bg_instances, patch, class_id):
    """Copy-Paste 增强：将小图块无缝克隆到背景图中"""
    H, W = bg_img.shape[:2]
    ph, pw = patch.shape[:2]
    
    new_img = bg_img.copy()
    new_instances = list(bg_instances)
    
    if pw >= W or ph >= H or pw <= 0 or ph <= 0:
        return new_img, new_instances
        
    max_x = W - pw
    max_y = H - ph
    if max_x <= 0 or max_y <= 0:
        return new_img, new_instances
        
    # 随机选择一个粘贴位置
    rx = random.randint(0, max_x)
    ry = random.randint(0, max_y)
    center = (rx + pw // 2, ry + ph // 2)
    
    try:
        # 尝试使用 OpenCV 的泊松融合 (无缝克隆) 让边缘过渡更自然
        mask = 255 * np.ones(patch.shape, patch.dtype)
        new_img = cv2.seamlessClone(patch, new_img, mask, center, cv2.NORMAL_CLONE)
    except:
        # 如果贴在边缘导致融合失败，回退到直接覆盖
        new_img[ry:ry+ph, rx:rx+pw] = patch
        
    # 计算新边界框的 YOLO 坐标
    nx = (rx + pw / 2.0) / W
    ny = (ry + ph / 2.0) / H
    nw = pw / float(W)
    nh = ph / float(H)
    
    new_instances.append((class_id, nx, ny, nw, nh))
    return new_img, new_instances

def main():
    print("⏳ 正在分析当前数据集分布...")
    dist, img_to_inst = get_current_distribution()
    
    print("\n📊 当前类别实例数:")
    for c in range(10):
        print(f"  - 类别 {c}: {dist.get(c, 0)} 个")
        
    print("\n✂️ 正在提取缺陷特征图块 (Patches Bank)...")
    patches = extract_patches(img_to_inst)
    all_images = list(img_to_inst.keys())
    
    aug_count = 0
    print(f"\n🚀 开始执行 Leakage-Aware 混合数据增强 (目标: {TARGET_INSTANCES}/类)...")
    
    # 动态记录当前分布
    current_dist = {k: v for k, v in dist.items()}
    
    for c in range(10):
        if current_dist[c] >= TARGET_INSTANCES:
            print(f"✅ 类别 {c} 样本充足 ({current_dist[c]} >= {TARGET_INSTANCES})，跳过。")
            continue
            
        # 根据漏检率计算两种策略的权重
        leakage = LEAKAGE_RATES.get(c, 0.5)
        cp_weight = leakage ** 2  # 漏检越严重，越依赖 Copy-Paste 增加背景多样性
        
        print(f"🔄 正在扩充类别 {c} (起始: {current_dist[c]}, 目标: {TARGET_INSTANCES}) | Copy-Paste 权重: {cp_weight:.0%} ...")
        
        c_images = [img for img, insts in img_to_inst.items() if any(i[0] == c for i in insts)]
        c_patches = patches.get(c, [])
        
        if not c_images and not c_patches:
            print(f"  ❌ 错误: 类别 {c} 没有任何数据，无法扩充！")
            continue
            
        # 动态判断是否达到了目标
        while current_dist[c] < TARGET_INSTANCES:
            choice = random.random()
            new_img = None
            new_insts = []
            
            # 选择 Copy-Paste (前提是有图块可贴)
            if choice < cp_weight and c_patches:
                # 优化：优先选择不包含太多其他类的背景，防止其他类爆炸
                bg_path = random.choice(all_images)
                bg_img = cv2.imread(bg_path)
                bg_insts = img_to_inst[bg_path]
                patch = random.choice(c_patches)
                new_img, new_insts = apply_copy_paste(bg_img, bg_insts, patch, c)
            
            # 选择基础图像处理 (前提是有原图可用)
            elif c_images:
                src_path = random.choice(c_images)
                src_img = cv2.imread(src_path)
                src_insts = img_to_inst[src_path]
                new_img, new_insts = apply_image_processing(src_img, src_insts)
                
            if new_img is not None:
                aug_count += 1
                new_name = f"aug_{c}_{aug_count:05d}"
                new_img_path = os.path.join(TRAIN_IMG_DIR, new_name + ".jpg")
                new_lbl_path = os.path.join(TRAIN_LBL_DIR, new_name + ".txt")
                
                cv2.imwrite(new_img_path, new_img)
                with open(new_lbl_path, "w") as f:
                    for inst in new_insts:
                        f.write(f"{inst[0]} {inst[1]:.6f} {inst[2]:.6f} {inst[3]:.6f} {inst[4]:.6f}\n")
                        # 动态更新所有涉及的类别数量
                        current_dist[inst[0]] += 1
                        
    print(f"\n🎉 增强完成！共生成了 {aug_count} 张合成图像及其标签。")
    
    print("\n📈 增强后类别实例数:")
    for c in range(10):
        print(f"  - 类别 {c}: {current_dist.get(c, 0)} 个")

if __name__ == "__main__":
    main()