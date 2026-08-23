import cv2
from ultralytics import YOLO
import config

class VisionAnalyzer:
    def __init__(self):
        # Allow ultralytics to auto-download if not present
        self.object_model = YOLO(config.OBJECT_MODEL_PATH)
        self.webcam_object_model = YOLO("models/yolov8n.pt")
        self.pose_model = YOLO(config.POSE_MODEL_PATH)
        
        # Move models to models/ dir if they downloaded to root
        import os, shutil
        for model_file in ["yolov8n.pt", "yolov8n-pose.pt"]:
            if os.path.exists(model_file) and not os.path.exists(f"models/{model_file}"):
                shutil.move(model_file, f"models/{model_file}")
                
    def process_frame(self, frame, imgsz_override=None, use_webcam_model=False):
        """
        Processes a single frame for objects (person, phone) and poses.
        Returns:
            dict containing lists of persons, phones, and poses
        """
        # Run object detection. Custom Model Classes: 0 is cell phone, 1 is person.
        inf_size = imgsz_override if imgsz_override else config.INFERENCE_IMGSZ
        model_to_use = self.webcam_object_model if use_webcam_model else self.object_model
        results_obj = model_to_use(frame, imgsz=inf_size, conf=config.PERSON_CONFIDENCE, verbose=False)
        
        persons = []
        phones = []
        
        if len(results_obj) > 0:
            boxes = results_obj[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                
                if use_webcam_model:
                    if cls_id == 67: cls_id = 0  # COCO cell phone -> Custom cell phone
                    elif cls_id == 0: cls_id = 1 # COCO person -> Custom person
                
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                
                if cls_id == 1:  # person
                    persons.append({'bbox': xyxy, 'conf': conf})
                elif cls_id == 0:  # cell phone
                    if conf >= config.PHONE_CONFIDENCE:
                        phones.append({'bbox': xyxy, 'conf': conf})
                        
        # Run pose estimation with tracking to enable per-seat baseline calibration
        results_pose = self.pose_model.track(frame, imgsz=inf_size, persist=True, tracker="bytetrack.yaml", conf=config.POSE_CONFIDENCE, verbose=False)
        
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
            'poses': poses
        }
