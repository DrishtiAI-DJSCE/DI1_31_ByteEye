from ultralytics import YOLO
import os

def evaluate_model():
    model_path = "models/drishti_detector.pt"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model {model_path} not found! Run training first.")
        return
        
    print("========================================")
    print("      DRISHTI MODEL EVALUATION")
    print("========================================")
    
    model = YOLO(model_path)
    yaml_path = os.path.abspath("dataset/data.yaml")
    
    # Evaluate on the test split
    print("[INFO] Evaluating on TEST set...")
    metrics = model.val(
        data=yaml_path,
        split='test',
        imgsz=960,
        conf=0.25, # Default confidence
        project="training_runs",
        name="evaluation_test_set"
    )
    
    print("\n----------------------------------------")
    print("          FINAL METRICS")
    print("----------------------------------------")
    print(f"Overall mAP50-95: {metrics.box.map:.4f}")
    print(f"Overall mAP50:    {metrics.box.map50:.4f}")
    
    # Per-class metrics
    names = metrics.names
    for i, c in enumerate(metrics.box.classes):
        cls_name = names[int(c)]
        # Map values are lists, we need the specific index
        # metrics.box.maps is an array where the index is the class index
        map50 = metrics.box.map50s[i] if hasattr(metrics.box, 'map50s') and len(metrics.box.map50s) > i else 0
        prec = metrics.box.p[i] if len(metrics.box.p) > i else 0
        rec = metrics.box.r[i] if len(metrics.box.r) > i else 0
        
        print(f"\nClass: {cls_name}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        
    print("========================================")

if __name__ == "__main__":
    evaluate_model()
