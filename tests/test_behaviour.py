import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from behaviour import BehaviourAnalyzer

def test_behaviour_persistence():
    analyzer = BehaviourAnalyzer()
    
    # Simulate phone globally
    # Frame 1: Phone appears
    result = analyzer.update("MOBILE_PHONE", True, 2.0)
    assert result == False
    
    # Frame 2: 1 second later
    time.sleep(1)
    result = analyzer.update("MOBILE_PHONE", True, 2.0)
    assert result == False
    
    # Frame 3: 2.1 seconds later (should trigger)
    time.sleep(1.2)
    result = analyzer.update("MOBILE_PHONE", True, 2.0)
    assert result == True # confirmed
    
    # Frame 4: immediately after (should be in cooldown, return False)
    result = analyzer.update("MOBILE_PHONE", True, 2.0)
    assert result == False
