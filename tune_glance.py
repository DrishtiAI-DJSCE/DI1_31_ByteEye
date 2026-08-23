import cv2
import glob
import os
from vision import VisionAnalyzer
from behaviour import BehaviourAnalyzer
from utils import get_keypoint

def analyze_dataset():
    vision = VisionAnalyzer()
    behaviour = BehaviourAnalyzer()
    
    images = glob.glob(r"dataset\test\images\*.jpg")
    print(f"Found {len(images)} test images.")
    
    ratios = []
    
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        res = vision.process_frame(frame)
        for pose in res['poses']:
            kpts = pose['keypoints']
            if len(kpts) < 7: continue
            nose_x, _, nose_c = get_keypoint(kpts[0])
            ls_x, _, ls_c = get_keypoint(kpts[5])
            rs_x, _, rs_c = get_keypoint(kpts[6])
            
            if nose_c > 0.35 and ls_c > 0.35 and rs_c > 0.35:
                shoulder_mid_x = (ls_x + rs_x) / 2.0
                shoulder_width = abs(ls_x - rs_x)
                if shoulder_width > 8.0:
                    ratio = abs(nose_x - shoulder_mid_x) / shoulder_width
                    ratios.append(ratio)
                    
    ratios.sort()
    n = len(ratios)
    if n == 0:
        print("No valid poses found.")
        return
        
    print(f"\nAnalyzed {n} poses.")
    print(f"Median ratio: {ratios[n//2]:.4f}")
    print(f"90th percentile: {ratios[int(n*0.90)]:.4f}")
    print(f"95th percentile: {ratios[int(n*0.95)]:.4f}")
    print(f"99th percentile: {ratios[int(n*0.99)]:.4f}")
    print(f"Max ratio: {ratios[-1]:.4f}")
    
    print("\nCandidate Deviations from 90th percentile baseline:")
    baseline = ratios[int(n*0.90)] # assume 90% are normal or slightly rotated
    baseline_median = ratios[n//2]
    
    for thresh in [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060]:
        fp = sum(1 for r in ratios if abs(r - baseline_median) > thresh)
        print(f"Threshold {thresh:.3f}: {fp} poses flagged ({fp/n*100:.1f}%)")

if __name__ == "__main__":
    analyze_dataset()
