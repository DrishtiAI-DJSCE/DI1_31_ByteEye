import time
import cv2
import config
from vision import VisionAnalyzer
from behaviour import BehaviourAnalyzer

def profile_run():
    print("Loading models...")
    vision = VisionAnalyzer()
    behaviour = BehaviourAnalyzer()
    
    # Try to open a default video or camera
    # We will just generate a dummy frame if no video exists
    print("Generating dummy frame for warmup...")
    import numpy as np
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(2):
        _ = vision.process_frame(dummy_frame)
        
    print("Starting profile on 10 iterations...")
    
    times = {'obj': [], 'pose': [], 'beh': []}
    
    for i in range(10):
        t0 = time.perf_counter()
        res = vision.process_frame(dummy_frame)
        t1 = time.perf_counter()
        
        _, _ = behaviour.analyze_frame_data(res['persons'], res['phones'], res['poses'], dummy_frame.shape)
        t2 = time.perf_counter()
        
        times['obj'].append(res['t_obj'] * 1000)
        times['pose'].append(res['t_pose'] * 1000)
        times['beh'].append((t2 - t1) * 1000)
        
    print("\n--- PROFILE RESULTS (ms) ---")
    print(f"YOLO Object: {sum(times['obj'])/10:.1f} ms")
    print(f"YOLO Pose:   {sum(times['pose'])/10:.1f} ms")
    print(f"Behaviour:   {sum(times['beh'])/10:.1f} ms")
    print(f"Total AI:    {(sum(times['obj']) + sum(times['pose']) + sum(times['beh']))/10:.1f} ms")
    print(f"Max AI FPS:  {1000 / ((sum(times['obj']) + sum(times['pose']) + sum(times['beh']))/10):.1f}")
    
    import torch
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
if __name__ == "__main__":
    profile_run()
