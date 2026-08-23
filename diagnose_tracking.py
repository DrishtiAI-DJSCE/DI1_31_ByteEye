import cv2
import glob
from vision import VisionAnalyzer
import config

def verify_tracking():
    # Use the test video if available, else we can't test tracking easily.
    # We can use the first 100 frames of an uploaded video.
    # Let's see what video is available.
    import os
    video_path = "test_video.mp4" # We need a video path. We can search for .mp4 files.
    videos = glob.glob("*.mp4")
    if not videos:
        print("No video found.")
        return
    
    video_path = videos[0]
    print(f"Testing tracking on {video_path}")
    
    vision = VisionAnalyzer()
    
    cap = cv2.VideoCapture(video_path)
    
    print("\n--- NO SKIPPING (30 FPS) ---")
    track_ids = set()
    for i in range(30):
        ret, frame = cap.read()
        if not ret: break
        res = vision.process_frame(frame)
        for p in res['poses']:
            if p['track_id']: track_ids.add(p['track_id'])
    print(f"Unique track IDs over 30 frames: {len(track_ids)}")
    
    cap.release()
    vision = VisionAnalyzer() # reset tracker
    cap = cv2.VideoCapture(video_path)
    
    print("\n--- SKIPPING 2 FRAMES (10 FPS) ---")
    track_ids = set()
    for i in range(10):
        ret, frame = cap.read()
        if not ret: break
        res = vision.process_frame(frame)
        for p in res['poses']:
            if p['track_id']: track_ids.add(p['track_id'])
            
        # Skip 2 frames
        cap.grab(); cap.grab()
        
    print(f"Unique track IDs over 10 frames (spanning same 30 source frames): {len(track_ids)}")

if __name__ == "__main__":
    verify_tracking()
