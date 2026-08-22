import os
import shutil
import csv

def fix_dataset():
    data_dir = "train_data"
    csv_path = os.path.join(data_dir, "dataset_audit.csv")
    
    # Read audit
    audit_results = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audit_results[row["filename"]] = row["verdict"]
            
    # Get all images
    old_train_images = os.path.join(data_dir, "train", "images")
    old_train_labels = os.path.join(data_dir, "train", "labels")
    
    images = sorted(os.listdir(old_train_images))
    total = len(images)
    
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    # Create new directories
    for split in ["train_new", "valid", "test"]:
        os.makedirs(os.path.join(data_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, split, "labels"), exist_ok=True)
        
    train_count = 0
    val_count = 0
    test_count = 0
    
    for i, img_name in enumerate(images):
        label_name = img_name.replace(".jpg", ".txt")
        verdict = audit_results.get(label_name, "OK")
        
        # Decide split sequentially
        if i < train_end:
            target_split = "train_new"
        elif i < val_end:
            target_split = "valid"
        else:
            target_split = "test"
            
        # Do not put NEEDS_REANNOTATION in valid or test
        if verdict == "NEEDS_REANNOTATION" and target_split in ["valid", "test"]:
            target_split = "train_new" # shove bad ground truth back to train so it doesn't pollute val/test metrics
            
        src_img = os.path.join(old_train_images, img_name)
        dst_img = os.path.join(data_dir, target_split, "images", img_name)
        
        src_lbl = os.path.join(old_train_labels, label_name)
        dst_lbl = os.path.join(data_dir, target_split, "labels", label_name)
        
        shutil.copy(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)
            
        if target_split == "train_new":
            train_count += 1
        elif target_split == "valid":
            val_count += 1
        else:
            test_count += 1

    print(f"Dataset restructured successfully.")
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    # Write new data.yaml
    yaml_content = f"""
path: {os.path.abspath(data_dir)}
train: train_new/images
val: valid/images
test: test/images

nc: 2
names: ['cell phone', 'person']
"""
    yaml_path = os.path.join(data_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print("Updated data.yaml")

if __name__ == "__main__":
    fix_dataset()
