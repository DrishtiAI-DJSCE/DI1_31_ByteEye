from ultralytics import YOLO
import config

print("=== PHASE 2: VERIFY THE MODEL'S CLASS MAPPING ===")
print(f"Loading model from: {config.OBJECT_MODEL_PATH}")
model = YOLO(config.OBJECT_MODEL_PATH)
print(f"model.names: {model.names}")
