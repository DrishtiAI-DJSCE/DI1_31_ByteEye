import cv2
import glob
from vision import VisionAnalyzer
from behaviour import BehaviourAnalyzer
import config
from unittest.mock import patch

def test_glance_threshold():
    images = glob.glob(r"dataset\test\images\*.jpg")
    print(f"Found {len(images)} test images.")
    
    config.INFERENCE_IMGSZ = 960
    vision = VisionAnalyzer()
    behaviour = BehaviourAnalyzer()
    
    # Lock baseline on the first image
    img_path = images[0]
    frame = cv2.imread(img_path)
    
    current_time = [0.0]
    def fake_time():
        return current_time[0]
        
    with patch('time.time', side_effect=fake_time):
        for i in range(60):
            current_time[0] = i * 0.1
            res = vision.process_frame(frame)
            behaviour.analyze_frame_data(res['persons'], res['phones'], res['poses'], frame.shape)
            
        print("\nTesting deviations on all test images against this baseline...")
        
        for i, img_path in enumerate(images[1:]):
            current_time[0] += 0.1
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            res = vision.process_frame(frame)
            poses = res['poses']
            
            for pose in poses:
                tid = pose.get('track_id')
                # we just need the nsr
                nsr = behaviour._nose_shoulder_ratio(pose['keypoints'])
                if nsr is not None:
                    # Let's find the closest baseline
                    # Since track IDs might change, we just compare against the locked baselines
                    deviations = []
                    for b_tid, tb in behaviour.track_baselines.items():
                        if tb['locked']:
                            deviations.append(abs(nsr - tb['baseline_yaw']))
                    if deviations:
                        min_dev = min(deviations)
                        print(f"Img {i}: nsr={nsr:.3f}, min_deviation={min_dev:.3f}")
            
if __name__ == "__main__":
    test_glance_threshold()
