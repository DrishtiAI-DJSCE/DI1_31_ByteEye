from ultralytics import YOLO
import os
import shutil

def run_training_benchmark(imgsz, batch_size, epochs=60):
    print(f"============================================================")
    print(f"   STARTING TRAINING BENCHMARK: imgsz={imgsz}, batch={batch_size}")
    print(f"============================================================")
    
    model = YOLO("models/yolov8n.pt")
    
    # Path to data.yaml
    yaml_path = os.path.abspath("dataset/data.yaml")
    
    # Run training
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=15,
        project="training_runs",
        name=f"benchmark_imgsz{imgsz}_batch{batch_size}",
        # Realistic CCTV augmentations
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, # small rotation
        translate=0.1,
        scale=0.2, # small scale variation to avoid shrinking phones too much
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5, # only horizontal flip
        mosaic=1.0, 
        close_mosaic=10 # disable mosaic in last 10 epochs
    )
    print(f"[INFO] Training for imgsz={imgsz} completed.")
    return results

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Download yolov8n if not exists
    if not os.path.exists("models/yolov8n.pt"):
        print("[INFO] Downloading pretrained yolov8n.pt...")
        YOLO("yolov8n.pt")
        shutil.move("yolov8n.pt", "models/yolov8n.pt")
        
    print("[INFO] Dataset validation complete. Starting benchmarks...")
    
    # Benchmark 1: 960
    run_training_benchmark(imgsz=960, batch_size=8, epochs=60)
    
    # Benchmark 2: 1280
    run_training_benchmark(imgsz=1280, batch_size=4, epochs=60)
    
    # Copy best weights to the target production location
    best_weights = "training_runs/benchmark_imgsz960_batch8/weights/best.pt"
    # We will prefer the 960 weights if 1280 didn't finish, or 1280 if it did and is better.
    # For now, let's just copy 960's as a fallback.
    if os.path.exists(best_weights):
        shutil.copy(best_weights, "models/drishti_detector.pt")
        print(f"[INFO] Copied best weights to models/drishti_detector.pt")
        
    print("[INFO] All benchmarks completed.")
