import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
source_dir = Path("GC10-DET")
lables_dir = source_dir / "lables"
output_dir = Path("GC10-DET_YOLO")

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Class mapping from Chinese Pinyin to English
CLASS_MAP = {
    '1_chongkong': 'punching_hole',
    '2_hanfeng': 'welding_line',
    '3_yueyawan': 'crescent_gap',
    '4_shuiban': 'water_spot',
    '5_youban': 'oil_spot',
    '6_siban': 'silk_spot',
    '7_yiwu': 'inclusion',
    '8_zhadang': 'rolled_pit',
    '8_yahen': 'rolled_pit',
    '9_zhehen': 'crease',
    '10_yaozhed': 'waist_folding',
    '10_yaozhe': 'waist_folding'
}

# The unique class names to build the index from (ensuring 10 classes)
UNIQUE_CLASSES = [
    'punching_hole', 'welding_line', 'crescent_gap', 'water_spot', 'oil_spot',
    'silk_spot', 'inclusion', 'rolled_pit', 'crease', 'waist_folding'
]

# Ensure deterministic split
random.seed(42)

def create_dirs():
    if output_dir.exists():
        print(f"Clearing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
        
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

def find_images():
    image_paths = {}
    # Search all subdirectories in source_dir for images
    for d in source_dir.iterdir():
        if d.is_dir() and d.name != "lables" and d.name != "datasets":
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                for img_path in d.glob(ext):
                    image_paths[img_path.name] = img_path
    return image_paths

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def main():
    print("Initializing directories...")
    create_dirs()
    
    print("Finding images...")
    image_paths = find_images()
    print(f"Found {len(image_paths)} images in subfolders.")
    
    class_names = UNIQUE_CLASSES
    
    valid_data = []
    
    print("Parsing XML annotations...")
    xml_files = list(lables_dir.glob("*.xml"))
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing {xml_file.name}: {e}")
            continue
        
        filename_element = root.find('filename')
        if filename_element is None:
            continue
            
        filename = filename_element.text
        img_name_stem = Path(filename).stem
        
        # Match image path
        matched_img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            possible_name = img_name_stem + ext
            if possible_name in image_paths:
                matched_img_path = image_paths[possible_name]
                break
                
        if not matched_img_path:
            continue
            
        size = root.find('size')
        if size is None:
            continue
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        if w == 0 or h == 0:
            continue
            
        yolo_labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in CLASS_MAP:
                print(f"Warning: Unknown class '{name}' in {xml_file.name}")
                continue
                
            cls_id = class_names.index(CLASS_MAP[name])
            
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            
            # Clip bounds and sanity check
            b = (max(0, b[0]), min(w, b[1]), max(0, b[2]), min(h, b[3]))
            if b[1] <= b[0] or b[3] <= b[2]:
                continue
                
            bb = convert_bbox((w, h), b)
            yolo_labels.append(f"{cls_id} {' '.join(map(str, bb))}")
            
        if yolo_labels: # Only append if there are valid bounding boxes
            valid_data.append({
                'img_path': matched_img_path,
                'labels': yolo_labels,
                'stem': img_name_stem
            })
            
    print(f"Found {len(valid_data)} valid annotations matching images.")
    
    # Shuffle and split
    random.shuffle(valid_data)
    num_total = len(valid_data)
    num_train = int(num_total * TRAIN_RATIO)
    num_val = int(num_total * VAL_RATIO)
    
    splits = {
        'train': valid_data[:num_train],
        'val': valid_data[num_train:num_train+num_val],
        'test': valid_data[num_train+num_val:]
    }
    
    # Copy files
    for split_name, data_list in splits.items():
        print(f"Writing {split_name} set ({len(data_list)} items)...")
        for item in data_list:
            out_img_path = output_dir / 'images' / split_name / item['img_path'].name
            out_lbl_path = output_dir / 'labels' / split_name / f"{item['stem']}.txt"
            
            shutil.copy(item['img_path'], out_img_path)
            
            with open(out_lbl_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(item['labels']) + '\n')
                
    # Create YAML
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {output_dir.absolute().as_posix()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
        
    print(f"Conversion complete! YOLO dataset saved at: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
