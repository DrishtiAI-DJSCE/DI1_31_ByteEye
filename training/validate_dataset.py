import os
import shutil
import random
import yaml

DATASET_DIR = "dataset"
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "train", "labels")

VAL_IMG_DIR = os.path.join(DATASET_DIR, "valid", "images")
VAL_LBL_DIR = os.path.join(DATASET_DIR, "valid", "labels")

TEST_IMG_DIR = os.path.join(DATASET_DIR, "test", "images")
TEST_LBL_DIR = os.path.join(DATASET_DIR, "test", "labels")

YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

def fix_dataset_split():
    """Splits the monolithic train/ directory into train, val, and test if needed."""
    if not os.path.exists(TRAIN_IMG_DIR):
        print("[ERROR] No train/images directory found!")
        return

    # If val/images already exists and has files, we assume split is done
    if os.path.exists(VAL_IMG_DIR) and len(os.listdir(VAL_IMG_DIR)) > 0:
        print("[INFO] Dataset already split. Skipping split phase.")
        return
        
    print("[INFO] Missing val/test splits. Generating 80-10-10 split from train...")
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LBL_DIR, exist_ok=True)
    os.makedirs(TEST_IMG_DIR, exist_ok=True)
    os.makedirs(TEST_LBL_DIR, exist_ok=True)

    images = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    val_count = int(len(images) * 0.1)
    test_count = int(len(images) * 0.1)
    
    val_images = images[:val_count]
    test_images = images[val_count:val_count+test_count]
    
    for img in val_images:
        base = os.path.splitext(img)[0]
        shutil.move(os.path.join(TRAIN_IMG_DIR, img), os.path.join(VAL_IMG_DIR, img))
        lbl = base + ".txt"
        if os.path.exists(os.path.join(TRAIN_LBL_DIR, lbl)):
            shutil.move(os.path.join(TRAIN_LBL_DIR, lbl), os.path.join(VAL_LBL_DIR, lbl))
            
    for img in test_images:
        base = os.path.splitext(img)[0]
        shutil.move(os.path.join(TRAIN_IMG_DIR, img), os.path.join(TEST_IMG_DIR, img))
        lbl = base + ".txt"
        if os.path.exists(os.path.join(TRAIN_LBL_DIR, lbl)):
            shutil.move(os.path.join(TRAIN_LBL_DIR, lbl), os.path.join(TEST_LBL_DIR, lbl))
            
    print(f"[INFO] Split complete. Moved {val_count} to val, {test_count} to test.")


def validate_dataset():
    """Generates a comprehensive dataset report."""
    print("========================================")
    print("       DATASET VALIDATION REPORT")
    print("========================================")

    # 1. Read YAML
    if not os.path.exists(YAML_PATH):
        print(f"[ERROR] {YAML_PATH} not found.")
        return
        
    with open(YAML_PATH, 'r') as f:
        data_yaml = yaml.safe_load(f)
        
    classes = data_yaml.get('names', [])
    print(f"Declared Classes ({len(classes)}): {classes}")
    
    # 2. Statistics tracking
    stats = {
        'total_images': 0,
        'missing_labels': 0,
        'empty_labels': 0,
        'class_counts': {i: 0 for i in range(len(classes))},
        'small_boxes': 0,
        'total_boxes': 0
    }
    
    # Analyze splits
    for split, img_dir, lbl_dir in [("Train", TRAIN_IMG_DIR, TRAIN_LBL_DIR), 
                                    ("Valid", VAL_IMG_DIR, VAL_LBL_DIR), 
                                    ("Test", TEST_IMG_DIR, TEST_LBL_DIR)]:
        if not os.path.exists(img_dir): continue
        imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        stats['total_images'] += len(imgs)
        print(f"{split} Images: {len(imgs)}")
        
        for img in imgs:
            lbl_file = os.path.join(lbl_dir, os.path.splitext(img)[0] + ".txt")
            if not os.path.exists(lbl_file):
                stats['missing_labels'] += 1
                continue
                
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
                
            if len(lines) == 0:
                stats['empty_labels'] += 1
                continue
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])
                    area = w * h
                    
                    if c in stats['class_counts']:
                        stats['class_counts'][c] += 1
                        
                    stats['total_boxes'] += 1
                    
                    # Box occupies less than 0.25% of frame (e.g. 64x64 in 1280x720)
                    if area < 0.0025:
                        stats['small_boxes'] += 1

    print("----------------------------------------")
    print(f"Total Images Analyzed: {stats['total_images']}")
    print(f"Images missing label files: {stats['missing_labels']}")
    print(f"Images with empty labels: {stats['empty_labels']}")
    print("----------------------------------------")
    for i, cls_name in enumerate(classes):
        print(f"Class '{cls_name}' instances: {stats['class_counts'].get(i, 0)}")
    print("----------------------------------------")
    print(f"Total Bounding Boxes: {stats['total_boxes']}")
    if stats['total_boxes'] > 0:
        small_pct = (stats['small_boxes'] / stats['total_boxes']) * 100
        print(f"Small Boxes (<0.25% area): {stats['small_boxes']} ({small_pct:.1f}%)")
        
        if small_pct > 20:
            print("[WARNING] High percentage of very small boxes detected. High-resolution training recommended.")
    print("========================================")

if __name__ == "__main__":
    fix_dataset_split()
    validate_dataset()
