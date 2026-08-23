import cv2
from ultralytics import YOLO
import config

class VisionAnalyzer:
    def __init__(self):
        # Allow ultralytics to auto-download if not present
        self.object_model = YOLO(config.OBJECT_MODEL_PATH)
        self.pose_model = YOLO(config.POSE_MODEL_PATH)
        
        # Move models to models/ dir if they downloaded to root
        import os, shutil
        for model_file in ["yolov8n.pt", "yolov8n-pose.pt"]:
            if os.path.exists(model_file) and not os.path.exists(f"models/{model_file}"):
                shutil.move(model_file, f"models/{model_file}")
                
    def process_frame(self, frame):
        """
        Processes a single frame for objects (person, phone) and poses.
        Returns:
            dict containing lists of persons, phones, and poses
        """
        import time
        
        t0 = time.perf_counter()
        # Run object detection. Custom Model Classes: 0 is cell phone, 1 is person.
        results_obj = self.object_model(frame, imgsz=config.INFERENCE_IMGSZ, conf=config.PERSON_CONFIDENCE, verbose=False)
        t1 = time.perf_counter()
        
        persons = []
        phones = []
        
        if len(results_obj) > 0:
            boxes = results_obj[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                
                if cls_id == 1:  # person
                    persons.append({'bbox': xyxy, 'conf': conf})
                elif cls_id == 0:  # cell phone
                    if conf >= config.PHONE_CONFIDENCE:
                        phones.append({'bbox': xyxy, 'conf': conf})
                        
        t2 = time.perf_counter()
        # Run pose estimation with tracking to enable per-seat baseline calibration
        results_pose = self.pose_model.track(frame, imgsz=config.INFERENCE_IMGSZ, persist=True, tracker="bytetrack.yaml", conf=config.POSE_CONFIDENCE, verbose=False)
        t3 = time.perf_counter()
        
        poses = []
        if len(results_pose) > 0:
            boxes = results_pose[0].boxes
            keypoints = results_pose[0].keypoints
            # Use .data to ensure we get [x, y, confidence] format if available
            if keypoints is not None and keypoints.data is not None and len(keypoints.data) > 0:
                for i in range(len(boxes)):
                    xyxy = boxes[i].xyxy[0].tolist()
                    kpts = keypoints.data[i].tolist() # guarantees [x,y,conf] shape if model provides it
                    track_id = int(boxes.id[i].item()) if boxes.id is not None else None
                    poses.append({
                        'bbox': xyxy,
                        'keypoints': kpts,
                        'track_id': track_id
                    })
                    
        return {
            'persons': persons,
            'phones': phones,
            'poses': poses,
            't_obj': t1 - t0,
            't_pose': t3 - t2
        }
