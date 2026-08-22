import os
import glob
import random
import cv2
import csv
import statistics

def audit_dataset():
    data_dir = "train_data"
    train_labels_dir = os.path.join(data_dir, "train", "labels")
    train_images_dir = os.path.join(data_dir, "train", "images")
    renders_dir = os.path.join(data_dir, "review_renders")
    
    os.makedirs(renders_dir, exist_ok=True)
    
    print("--- 1. Structural Checks ---")
    valid_exists = os.path.isdir(os.path.join(data_dir, "valid", "images"))
    test_exists = os.path.isdir(os.path.join(data_dir, "test", "images"))
    print(f"valid/images exists: {valid_exists}")
    print(f"test/images exists: {test_exists}")
    
    label_files = glob.glob(os.path.join(train_labels_dir, "*.txt"))
    print(f"\nFound {len(label_files)} label files in {train_labels_dir}")
    
    phone_areas = []
    person_areas = []
    
    audit_rows = []
    
    images_with_phones = []
    
    print("\n--- 2. Parsing Labels ---")
    for lf in label_files:
        basename = os.path.basename(lf)
        img_name = basename.replace(".txt", ".jpg")
        
        has_phone = False
        img_phone_areas = []
        img_person_areas = []
        
        with open(lf, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    # w is parts[3], h is parts[4]
                    w = float(parts[3])
                    h = float(parts[4])
                    area_frac = w * h
                    
                    if cls == 0:
                        has_phone = True
                        phone_areas.append(area_frac)
                        img_phone_areas.append(area_frac)
                    elif cls == 1:
                        person_areas.append(area_frac)
                        img_person_areas.append(area_frac)
                        
        if has_phone:
            images_with_phones.append(basename)
            
        # Verdict Logic
        verdict = "OK"
        if len(img_phone_areas) == 0 and len(img_person_areas) == 0:
            verdict = "EMPTY"
        else:
            # If any phone box is > 5% of the frame, flag it
            for a in img_phone_areas:
                if a > 0.05:
                    verdict = "NEEDS_REANNOTATION"
                    break
                    
        audit_rows.append({
            "filename": basename,
            "verdict": verdict,
            "phone_areas": str([round(a, 4) for a in img_phone_areas]),
            "person_areas": str([round(a, 4) for a in img_person_areas])
        })
        
    print("\n--- 3. Label Statistics ---")
    
    def print_stats(name, areas):
        if not areas:
            print(f"{name}: NO BOXES FOUND")
            return
        
        count = len(areas)
        mean_area = statistics.mean(areas)
        median_area = statistics.median(areas)
        min_area = min(areas)
        max_area = max(areas)
        gt_5 = sum(1 for a in areas if a > 0.05) / count * 100
        gt_15 = sum(1 for a in areas if a > 0.15) / count * 100
        
        print(f"[{name}]")
        print(f"Count: {count}")
        print(f"Mean Area Fraction: {mean_area:.4f} ({mean_area*100:.2f}%)")
        print(f"Median Area Fraction: {median_area:.4f} ({median_area*100:.2f}%)")
        print(f"Min: {min_area:.4f}, Max: {max_area:.4f}")
        print(f"Exceeds 5% of frame: {gt_5:.1f}%")
        print(f"Exceeds 15% of frame: {gt_15:.1f}%")
        print()

    print_stats("Cell Phone (Class 0)", phone_areas)
    print_stats("Person (Class 1)", person_areas)
    
    print("\n--- 4. Rendering Samples ---")
    random.shuffle(images_with_phones)
    sample_to_render = images_with_phones[:10]
    
    for lf_name in sample_to_render:
        lf_path = os.path.join(train_labels_dir, lf_name)
        img_name = lf_name.replace(".txt", ".jpg")
        img_path = os.path.join(train_images_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        with open(lf_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    bx, by, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    
                    if cls == 0:
                        color = (0, 0, 255) # Red for phone
                    else:
                        color = (255, 0, 0) # Blue for person
                        
                    x1 = int((bx - bw / 2) * w)
                    y1 = int((by - bh / 2) * h)
                    x2 = int((bx + bw / 2) * w)
                    y2 = int((by + bh / 2) * h)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
        render_path = os.path.join(renders_dir, img_name)
        cv2.imwrite(render_path, img)
    print(f"Rendered {len(sample_to_render)} sample images to {renders_dir}")
    
    print("\n--- 5. Generating Audit CSV ---")
    csv_path = os.path.join(data_dir, "dataset_audit.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "verdict", "phone_areas", "person_areas"])
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"Audit CSV written to {csv_path}")

if __name__ == "__main__":
    audit_dataset()
