import time
import config
from severity import get_severity

class BehaviourAnalyzer:
    def __init__(self):
        # state is tracked per event_type globally to prevent multiple instances of the same event
        self.state = {}

    def _get_state_key(self, event_type):
        return event_type

    def update(self, event_type, is_occurring, persistence_threshold):
        key = self._get_state_key(event_type)
        now = time.time()

        if key not in self.state:
            self.state[key] = {"first_seen": 0, "last_seen": 0, "active": False, "cooldown_until": 0}

        st = self.state[key]

        if now < st["cooldown_until"]:
            return False, st["active"] # Return newly_confirmed, is_currently_active

        newly_confirmed = False
        if is_occurring:
            if st["first_seen"] == 0:
                st["first_seen"] = now
            st["last_seen"] = now

            if (now - st["first_seen"]) >= persistence_threshold and not st["active"]:
                st["active"] = True
                st["cooldown_until"] = now + config.EVENT_COOLDOWN_SECONDS
                newly_confirmed = True
        else:
            if (now - st["last_seen"]) > 1.0:
                 st["first_seen"] = 0
                 st["active"] = False
                 
        return newly_confirmed, st["active"]
        
    def analyze_frame_data(self, persons, phones, poses):
        """
        Analyzes the detections and poses for behaviours.
        Returns a tuple: (confirmed_events, active_highlights)
        """
        confirmed_events = []
        active_highlights = []
        now_ts = time.time()
        
        # 1. Process Phones globally
        is_phone_present = len(phones) > 0
        phone_confirmed, phone_active = self.update("MOBILE_PHONE", is_phone_present, config.PHONE_PERSISTENCE_SECONDS)
        best_phone = max(phones, key=lambda p: p['conf']) if is_phone_present else None
        
        if phone_confirmed:
            confirmed_events.append({
                "event_type": "MOBILE_PHONE",
                "confidence": best_phone['conf'] if best_phone else 1.0,
                "bbox": best_phone['bbox'] if best_phone else None,
                "timestamp": now_ts,
                "severity": get_severity("MOBILE_PHONE")
            })
            
        if phone_active and best_phone:
            active_highlights.append({
                "bbox": best_phone['bbox'],
                "severity": get_severity("MOBILE_PHONE"),
                "label": "MOBILE PHONE"
            })
        
        # 2. Process Persons/Poses for glance/rotation
        is_turning = False
        is_rotating = False
        best_pose_turn = None
        best_pose_rot = None
        
        for pose in poses:
            kpts = pose['keypoints']
            if len(kpts) > 4:
                le_x, le_y, le_c = kpts[3]
                re_x, re_y, re_c = kpts[4]
                if (le_c > 0.7 and re_c < 0.3) or (re_c > 0.7 and le_c < 0.3):
                    is_turning = True
                    best_pose_turn = pose
                    
            if len(kpts) > 6:
                ls_x, ls_y, ls_c = kpts[5]
                rs_x, rs_y, rs_c = kpts[6]
                if (ls_c > 0.7 and rs_c < 0.3) or (rs_c > 0.7 and ls_c < 0.3):
                    is_rotating = True
                    best_pose_rot = pose
            
        turn_confirmed, turn_active = self.update("SIDEWARD_GLANCE", is_turning, config.HEAD_TURN_PERSISTENCE_SECONDS)
        rot_confirmed, rot_active = self.update("BODY_ROTATION", is_rotating, config.BODY_ROTATION_PERSISTENCE_SECONDS)
        
        if turn_confirmed:
            confirmed_events.append({
                "event_type": "SIDEWARD_GLANCE",
                "confidence": 1.0,
                "bbox": best_pose_turn['bbox'] if best_pose_turn else None,
                "timestamp": now_ts,
                "severity": get_severity("SIDEWARD_GLANCE")
            })
            
        if turn_active and best_pose_turn:
            active_highlights.append({
                "bbox": best_pose_turn['bbox'],
                "severity": get_severity("SIDEWARD_GLANCE"),
                "label": "SIDEWARD GLANCE"
            })
                
        if rot_confirmed:
            confirmed_events.append({
                "event_type": "BODY_ROTATION",
                "confidence": 1.0,
                "bbox": best_pose_rot['bbox'] if best_pose_rot else None,
                "timestamp": now_ts,
                "severity": get_severity("BODY_ROTATION")
            })
            
        if rot_active and best_pose_rot:
            active_highlights.append({
                "bbox": best_pose_rot['bbox'],
                "severity": get_severity("BODY_ROTATION"),
                "label": "BODY ROTATION"
            })
                
        return confirmed_events, active_highlights
