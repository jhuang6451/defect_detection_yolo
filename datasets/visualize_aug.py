import os
import cv2
import glob
import random

# 配置路径
IMG_DIR = "datasets/GC10-DET-YOLO-Augmented/images/train"
LBL_DIR = "datasets/GC10-DET-YOLO-Augmented/labels/train"
SAVE_DIR = "datasets/GC10-DET-YOLO-Augmented/aug_check"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 类别名称
NAMES = ['punching_hole', 'welding_line', 'crescent_gap', 'water_spot', 'oil_spot', 'silk_spot', 'inclusion', 'rolled_pit', 'crease', 'waist_folding']

def visualize():
    # 筛选出增强生成的文件
    aug_images = glob.glob(os.path.join(IMG_DIR, "aug_*.jpg"))
    if not aug_images:
        print("未找到 aug_ 开头的增强图片，请检查路径。")
        return
    
    # 随机抽 10 张
    samples = random.sample(aug_images, min(10, len(aug_images)))
    print(f"正在从 {len(aug_images)} 张增强图中抽取 10 张进行可视化验证...")

    for img_path in samples:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        lbl_path = os.path.join(LBL_DIR, os.path.basename(img_path).replace(".jpg", ".txt"))
        if not os.path.exists(lbl_path): continue
        
        with open(lbl_path, "r") as f:
            for line in f:
                c, x, y, nw, nh = map(float, line.strip().split())
                c = int(c)
                
                # 转换回像素坐标
                x1 = int((x - nw/2) * w)
                y1 = int((y - nh/2) * h)
                x2 = int((x + nw/2) * w)
                y2 = int((y + nh/2) * h)
                
                # 画框和类别名
                color = (0, 255, 0) # 绿色
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{NAMES[c]}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        save_path = os.path.join(SAVE_DIR, os.path.basename(img_path))
        cv2.imwrite(save_path, img)
        print(f"已保存验证图: {save_path}")

if __name__ == "__main__":
    visualize()
