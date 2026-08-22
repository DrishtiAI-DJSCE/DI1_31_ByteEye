import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from behaviour import BehaviourAnalyzer
from utils import get_keypoint

def test_behaviour_safe_unpacking_2_elements():
    analyzer = BehaviourAnalyzer()
    
    # 2 elements per keypoint (no confidence)
    kpts_2 = [[100, 100] for _ in range(17)]
    poses = [{'bbox': [0,0,10,10], 'keypoints': kpts_2}]
    
    # Should not crash
    confirmed, active = analyzer.analyze_frame_data([], [], poses)
    assert len(confirmed) == 0
    
def test_behaviour_safe_unpacking_3_elements():
    analyzer = BehaviourAnalyzer()
    
    # 3 elements per keypoint (with confidence)
    kpts_3 = [[100, 100, 0.95] for _ in range(17)]
    poses = [{'bbox': [0,0,10,10], 'keypoints': kpts_3}]
    
    # Should not crash
    confirmed, active = analyzer.analyze_frame_data([], [], poses)
    assert len(confirmed) == 0
    
def test_utils_get_keypoint():
    x, y, c = get_keypoint([10, 20])
    assert x == 10
    assert y == 20
    assert c is None
    
    x, y, c = get_keypoint([10, 20, 0.5])
    assert x == 10
    assert y == 20
    assert c == 0.5
