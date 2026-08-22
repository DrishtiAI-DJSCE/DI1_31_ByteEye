import cv2
import numpy as np
from vision import VisionAnalyzer

def test_vision():
    print("Initializing VisionAnalyzer...")
    vision = VisionAnalyzer()
    
    print("Creating dummy frame...")
    # Create a dummy image (e.g., 640x480 black image)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Processing frame...")
    results = vision.process_frame(frame)
    
    print("Results:")
    print("Persons:", len(results['persons']))
    print("Phones:", len(results['phones']))
    print("Poses:", len(results['poses']))
    print("Vision test completed successfully.")

if __name__ == "__main__":
    test_vision()
