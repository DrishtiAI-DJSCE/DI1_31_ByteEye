import cv2
import os
import shutil
from ultralytics import YOLO

def main():
    video_path = os.path.join("data", "01.Candidate was found using a mobile phone in the examination hall.mkv")
    output_dir = "dataset"
    raw_frames_dir = os.path.join(output_dir, "raw_frames")
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    for d in [raw_frames_dir, images_dir, labels_dir]:
        os.makedirs(d, exist_ok=True)

    print("Loading stock YOLOv8n model for pre-annotation...")
    model = YOLO("models/yolov8n.pt")
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    frame_count = 0
    saved_count = 0
    frames_with_phone = 0
    frames_no_phone = 0
    
    frame_paths = []
    
    sample_rate = 15 # extract 1 frame every 15 frames

    print("Extracting frames and pre-annotating...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_rate == 0:
            frame_name = f"frame_{saved_count:06d}"
            image_path = os.path.join(raw_frames_dir, f"{frame_name}.jpg")
            
            # Save raw frame
            cv2.imwrite(image_path, frame)
            frame_paths.append(frame_name)
            
            # Pre-annotate
            results = model.predict(frame, verbose=False, classes=[67]) # Class 67 is cell phone in COCO
            
            label_path = os.path.join(labels_dir, f"{frame_name}.txt")
            
            has_phone = False
            with open(label_path, 'w') as f:
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Remap to class 0
                        cls = 0
                        x, y, w, h = box.xywhn[0].tolist()
                        f.write(f"{cls} {x} {y} {w} {h}\n")
                        has_phone = True
            
            if has_phone:
                frames_with_phone += 1
            else:
                frames_no_phone += 1
                
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"Processed {saved_count} frames...")

        frame_count += 1

    cap.release()
    print(f"\n--- DATA EXTRACTION SUMMARY ---")
    print(f"Total extracted frames: {saved_count}")
    print(f"Frames with >= 1 pre-annotated phone: {frames_with_phone}")
    print(f"Frames with NO phone detected: {frames_no_phone}")
    print(f"-------------------------------\n")

    print("Splitting dataset into train/val/test (80/10/10)...")
    
    # Organize into train/val/test
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    for split in splits.keys():
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
        
    num_frames = len(frame_paths)
    train_end = int(num_frames * splits["train"])
    val_end = train_end + int(num_frames * splits["val"])
    
    for i, frame_name in enumerate(frame_paths):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"
            
        # Move image
        src_img = os.path.join(raw_frames_dir, f"{frame_name}.jpg")
        dst_img = os.path.join(images_dir, split, f"{frame_name}.jpg")
        shutil.copy(src_img, dst_img)
        
        # Move label
        src_lbl = os.path.join(labels_dir, f"{frame_name}.txt")
        dst_lbl = os.path.join(labels_dir, split, f"{frame_name}.txt")
        shutil.copy(src_lbl, dst_lbl)
        
        # Delete original label from base dir to clean up
        os.remove(src_lbl)
        
    print("Writing data.yaml...")
    yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['phone']
"""
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)
        
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
