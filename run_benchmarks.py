import time
import numpy as np
from vision import VisionAnalyzer
import config

def run_benchmarks():
    print("Loading models...")
    vision = VisionAnalyzer()
    
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    print("\n--- BENCHMARK MATRIX ---")
    
    for imgsz in [640, 960, 1280]:
        config.INFERENCE_IMGSZ = imgsz
        
        # Warmup
        for _ in range(2):
            _ = vision.process_frame(dummy_frame)
            
        times = {'obj': [], 'pose': []}
        
        for i in range(10):
            res = vision.process_frame(dummy_frame)
            times['obj'].append(res['t_obj'] * 1000)
            times['pose'].append(res['t_pose'] * 1000)
            
        t_obj = sum(times['obj'])/10
        t_pose = sum(times['pose'])/10
        t_total = t_obj + t_pose
        fps = 1000 / t_total if t_total > 0 else 0
        
        print(f"imgsz={imgsz}: Obj={t_obj:.1f}ms, Pose={t_pose:.1f}ms, Total={t_total:.1f}ms -> FPS: {fps:.1f}")

if __name__ == "__main__":
    run_benchmarks()
