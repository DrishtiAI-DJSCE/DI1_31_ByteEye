import time
import collections
import config
from severity import get_severity
from utils import get_keypoint, calculate_yaw_ratio, is_hand_raised

class BehaviourAnalyzer:
    def __init__(self):
        # state is tracked per event_type globally to prevent multiple instances of the same event
        self.state = {}
        self.episode_history = collections.defaultdict(collections.deque)
        
    def reset(self):
        """Reset the internal state for a new session."""
        self.state = {}
        self.episode_history = collections.defaultdict(collections.deque)

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
                 if st["first_seen"] != 0:
                     duration = st["last_seen"] - st["first_seen"]
                     self.episode_history[key].append((st["last_seen"], duration))
                 st["first_seen"] = 0
                 st["active"] = False
                 
        return newly_confirmed, st["active"]
        
    def check_frequency_anomaly(self, event_type, window_seconds, min_count):
        """Checks if there have been min_count episodes within window_seconds."""
        now = time.time()
        history = self.episode_history[event_type]
        
        # Prune old episodes
        while history and (now - history[0][0]) > window_seconds:
            history.popleft()
            
        if len(history) >= min_count:
            # Clear history to avoid rapid re-triggering (or rely on cooldown)
            history.clear()
            # Also put the main tracker on cooldown so we don't spam
            if event_type in self.state:
                self.state[event_type]["cooldown_until"] = now + config.EVENT_COOLDOWN_SECONDS
            return True
            
        return False
        
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
        
        # 2. Process Persons/Poses for glance/rotation/hand-raise
        is_turning = False
        is_rotating = False
        is_hand_up = False
        
        best_pose_turn = None
        best_pose_rot = None
        best_pose_hand = None
        
        for pose in poses:
            kpts = pose['keypoints']
            bbox = pose['bbox']
            bbox_height = abs(bbox[3] - bbox[1])
            
            # SIDEWARD GLANCE
            if len(kpts) > 4:
                n_kpt = get_keypoint(kpts[0])
                le_kpt = get_keypoint(kpts[3])
                re_kpt = get_keypoint(kpts[4])
                
                n_c, le_c, re_c = n_kpt[2], le_kpt[2], re_kpt[2]
                
                if le_c is not None and re_c is not None:
                    # Rule 1: Confidence-collapse
                    if (le_c > 0.7 and re_c < 0.3) or (re_c > 0.7 and le_c < 0.3):
                        is_turning = True
                        best_pose_turn = pose
                    # Rule 2: Yaw-Ratio (gated by confidence)
                    elif le_c > 0.3 and re_c > 0.3 and n_c is not None and n_c > 0.3:
                        yaw_ratio = calculate_yaw_ratio(n_kpt, le_kpt, re_kpt)
                        if yaw_ratio > config.HEAD_TURN_THRESHOLD_RATIO:
                            is_turning = True
                            best_pose_turn = pose
                            
            # BODY ROTATION
            if len(kpts) > 6:
                ls_kpt = get_keypoint(kpts[5])
                rs_kpt = get_keypoint(kpts[6])
                
                ls_c, rs_c = ls_kpt[2], rs_kpt[2]
                
                if ls_c is not None and rs_c is not None:
                    # Rule 1: Confidence-collapse
                    if (ls_c > 0.7 and rs_c < 0.3) or (rs_c > 0.7 and ls_c < 0.3):
                        is_rotating = True
                        best_pose_rot = pose
                    # Rule 2: Shoulder Yaw-Ratio (relative to nose)
                    # Gate this by `not is_turning` to prevent nose-contamination from independent head turns
                    elif ls_c > 0.3 and rs_c > 0.3 and n_kpt[2] is not None and n_kpt[2] > 0.3 and not is_turning:
                        shoulder_ratio = calculate_yaw_ratio(n_kpt, ls_kpt, rs_kpt)
                        if shoulder_ratio > config.BODY_ROTATION_THRESHOLD_RATIO:
                            is_rotating = True
                            best_pose_rot = pose
                            
            # HAND RAISE (HAND_MOVEMENT)
            if is_hand_raised(kpts, bbox_height, config.HAND_MOVEMENT_THRESHOLD):
                is_hand_up = True
                best_pose_hand = pose
            
        turn_confirmed, turn_active = self.update("SIDEWARD_GLANCE", is_turning, config.HEAD_TURN_PERSISTENCE_SECONDS)
        turn_freq = self.check_frequency_anomaly("SIDEWARD_GLANCE", 15.0, 3)
        
        rot_confirmed, rot_active = self.update("BODY_ROTATION", is_rotating, config.BODY_ROTATION_PERSISTENCE_SECONDS)
        rot_freq = self.check_frequency_anomaly("BODY_ROTATION", 15.0, 3)
        
        hand_confirmed, hand_active = self.update("HAND_MOVEMENT", is_hand_up, config.HAND_MOVEMENT_PERSISTENCE_SECONDS)
        
        if turn_confirmed or turn_freq:
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
                
        if rot_confirmed or rot_freq:
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
            
        if hand_confirmed:
            confirmed_events.append({
                "event_type": "HAND_MOVEMENT",
                "confidence": 1.0,
                "bbox": best_pose_hand['bbox'] if best_pose_hand else None,
                "timestamp": now_ts,
                "severity": get_severity("HAND_MOVEMENT")
            })
            
        if hand_active and best_pose_hand:
            active_highlights.append({
                "bbox": best_pose_hand['bbox'],
                "severity": get_severity("HAND_MOVEMENT"),
                "label": "HAND RAISE"
            })
                
        return confirmed_events, active_highlights
