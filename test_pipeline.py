import cv2
import config
from vision import VisionAnalyzer
from behaviour import BehaviourAnalyzer

# 1. Load model and verify names
vision = VisionAnalyzer()
print("Model names:", vision.object_model.names)

# 2. Read image
img_path = r"dataset\test\images\03-CCTV Mobile Usage_000064_jpg.rf.PXelmvtfWQ2nBzqzfqCb.jpg"
frame = cv2.imread(img_path)
if frame is None:
    print("Could not load image.")
    exit(1)

# 3. Process frame (run 10 times to simulate temporal persistence and cooldown)
print("\n--- Running 10 simulated frames ---")
behaviour = BehaviourAnalyzer()
for i in range(10):
    res = vision.process_frame(frame)
    # Mock an UNASSOCIATED phone for frames 0-5
    if i < 6:
        # placing it far away from the person
        res['phones'].append({'bbox': [800, 800, 850, 850], 'conf': 0.85})
    
    if i == 0:
        print(f"Frame {i}: Found {len(res['persons'])} persons, {len(res['phones'])} phones, {len(res['poses'])} poses.")
    
    events, highlights = behaviour.analyze_frame_data(res['persons'], res['phones'], res['poses'], frame.shape)
    for e in events:
        print(f"EVENT at frame {i}: {e['event_type']} (Severity: {e.get('severity')})")
    for h in highlights:
        pass # print(f"HIGHLIGHT at frame {i}: {h['label']}")

print("\nPipeline test complete. No crashes!")
