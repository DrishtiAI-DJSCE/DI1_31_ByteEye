import os
import time
from ultralytics import YOLO

# Callback state variables
epoch_start_time = 0

def on_train_epoch_start(trainer):
    global epoch_start_time
    epoch_start_time = time.time()

def on_fit_epoch_end(trainer):
    global epoch_start_time
    epoch = trainer.epoch + 1
    total_epochs = trainer.epochs
    
    elapsed = time.time() - epoch_start_time
    eta = elapsed * (total_epochs - epoch)
    
    metrics = trainer.metrics
    # Ultralytics metrics keys can be tricky, typically format is like:
    # metrics['metrics/mAP50(B)'] or just checking trainer.validator.metrics
    
    # We will safely pull the values
    box_loss = trainer.loss.item() if hasattr(trainer, 'loss') else 0.0
    
    results = trainer.metrics
    p = results.get('metrics/precision(B)', 0.0)
    r = results.get('metrics/recall(B)', 0.0)
    map50 = results.get('metrics/mAP50(B)', 0.0)
    map50_95 = results.get('metrics/mAP50-95(B)', 0.0)
    
    print(f"\n[LIVE PROGRESS] Epoch {epoch}/{total_epochs} | Time: {elapsed:.1f}s (ETA: {eta:.1f}s)")
    print(f"Loss: {box_loss:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}\n")


def main():
    print("--- DIAGNOSTIC TEST RUN STARTING ---")
    print("Note: This run is intended to test the failure mode of oversized phone boxes.")
    
    yaml_path = os.path.abspath(os.path.join("train_data", "data.yaml"))
    
    model = YOLO("models/yolov8n.pt")
    
    # Attach callbacks
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # Train
    results = model.train(
        data=yaml_path,
        epochs=25,
        imgsz=1280,
        device='0',
        project='train_data/models',
        name='roboflow_test',
        exist_ok=True,
        workers=0,
        batch=-1,
        amp=True
    )
    
    print("\n--- DIAGNOSTIC RUN COMPLETE ---")
    print("Final Model Validation Metrics (Overall):")
    print(f"mAP50: {results.box.map50:.4f}")
    
    print("\nAttempting class-specific metrics breakdown:")
    try:
        # Evaluate on test set
        print("Running on test set...")
        test_metrics = model.val(split='test')
        
        print("\n=== TEST SET METRICS (CLASS-SPECIFIC) ===")
        # Class 0 = cell phone, Class 1 = person
        # Maps usually store class-wise data in .class_result
        if hasattr(test_metrics.box, 'class_result'):
            for i, c in enumerate(test_metrics.box.ap_class_index):
                cls_name = test_metrics.names[c]
                p = test_metrics.box.p[i]
                r = test_metrics.box.r[i]
                map50 = test_metrics.box.map50[i]
                print(f"Class '{cls_name}': Precision={p:.4f}, Recall={r:.4f}, mAP50={map50:.4f}")
        else:
            print("Class-specific results structure not found in metrics object.")
            print(f"Overall Test mAP50: {test_metrics.box.map50:.4f}")
            
    except Exception as e:
        print(f"Could not run test set evaluation: {e}")

if __name__ == "__main__":
    main()
