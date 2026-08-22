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
        self.track_baselines = {}
        
    def reset(self):
        """Reset the internal state for a new session."""
        self.state = {}
        self.episode_history = collections.defaultdict(collections.deque)
        self.track_baselines = {}

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
        
    def process_calibration(self, track_id, raw_yaw, shoulder_width, now_ts):
        """
        Updates the per-seat calibration buffer for the given track_id.
        Returns a tuple: (baseline_yaw, baseline_shoulder_width) if calibrated, else (None, None).
        """
        if track_id is None:
            return None, None
            
        if track_id not in self.track_baselines:
            self.track_baselines[track_id] = {
                "start_time": now_ts,
                "history_yaw": [],
                "history_width": [],
                "locked": False,
                "baseline_yaw": None,
                "baseline_width": None,
                "last_seen": now_ts
            }
            
        tb = self.track_baselines[track_id]
        tb["last_seen"] = now_ts
        
        if tb["locked"]:
            return tb["baseline_yaw"], tb["baseline_width"]
            
        # Add to history if we have valid readings
        if raw_yaw is not None:
            tb["history_yaw"].append(raw_yaw)
        if shoulder_width is not None:
            tb["history_width"].append(shoulder_width)
            
        # Keep window size reasonable (max 150 frames ~ 10 seconds at 15fps)
        if len(tb["history_yaw"]) > 150:
            tb["history_yaw"].pop(0)
        if len(tb["history_width"]) > 150:
            tb["history_width"].pop(0)
            
        # Check if we should lock
        if (now_ts - tb["start_time"]) >= config.CALIBRATION_SECONDS:
            if len(tb["history_yaw"]) > 20 and len(tb["history_width"]) > 20:
                yaw_variance = max(tb["history_yaw"]) - min(tb["history_yaw"])
                # If stable (variance < 0.15)
                if yaw_variance < 0.15:
                    sorted_yaw = sorted(tb["history_yaw"])
                    sorted_width = sorted(tb["history_width"])
                    tb["baseline_yaw"] = sorted_yaw[len(sorted_yaw)//2]
                    tb["baseline_width"] = sorted_width[len(sorted_width)//2]
                    tb["locked"] = True
                    return tb["baseline_yaw"], tb["baseline_width"]
            
            # If not stable, slide window forward
            tb["start_time"] = now_ts - (config.CALIBRATION_SECONDS / 2.0)
            
        return None, None
        
    def analyze_frame_data(self, persons, phones, poses):
        """
        Analyzes the detections and poses for behaviours.
        Returns a tuple: (confirmed_events, active_highlights)
        """
        confirmed_events = []
        active_highlights = []
        now_ts = time.time()
        
        # Prune stale tracks
        stale_ids = [tid for tid, tb in self.track_baselines.items() if (now_ts - tb["last_seen"]) > 5.0]
        for tid in stale_ids:
            del self.track_baselines[tid]
        
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
            
            # Extract raw features if visible
            raw_yaw = None
            shoulder_width = None
            
            n_kpt = get_keypoint(kpts[0]) if len(kpts) > 0 else (0,0,None)
            n_c = n_kpt[2]
            
            le_c, re_c, ls_c, rs_c = None, None, None, None
            
            if len(kpts) > 4:
                le_kpt = get_keypoint(kpts[3])
                re_kpt = get_keypoint(kpts[4])
                le_c, re_c = le_kpt[2], re_kpt[2]
                if le_c is not None and re_c is not None and le_c > 0.3 and re_c > 0.3 and n_c is not None and n_c > 0.3:
                    raw_yaw = calculate_yaw_ratio(n_kpt, le_kpt, re_kpt)
                    
            if len(kpts) > 6:
                ls_kpt = get_keypoint(kpts[5])
                rs_kpt = get_keypoint(kpts[6])
                ls_c, rs_c = ls_kpt[2], rs_kpt[2]
                if ls_c is not None and rs_c is not None and ls_c > 0.3 and rs_c > 0.3:
                    shoulder_width = abs(ls_kpt[0] - rs_kpt[0])
                    
            # Process Calibration
            track_id = pose.get('track_id')
            baseline_yaw, baseline_width = self.process_calibration(track_id, raw_yaw, shoulder_width, now_ts)
            
            # SIDEWARD GLANCE
            if len(kpts) > 4:
                # Rule 1: Confidence collapse override
                if le_c is not None and re_c is not None:
                    if (le_c > 0.7 and re_c < 0.3) or (re_c > 0.7 and le_c < 0.3):
                        is_turning = True
                        best_pose_turn = pose
                
                # Rule 2: Deviation from baseline
                if baseline_yaw is not None and raw_yaw is not None:
                    yaw_deviation = abs(raw_yaw - baseline_yaw)
                    if yaw_deviation > config.YAW_DEVIATION_THRESHOLD:
                        is_turning = True
                        best_pose_turn = pose
                        
            # BODY ROTATION
            if len(kpts) > 6:
                # Rule 1: Confidence collapse override
                if ls_c is not None and rs_c is not None:
                    if (ls_c > 0.7 and rs_c < 0.3) or (rs_c > 0.7 and ls_c < 0.3):
                        is_rotating = True
                        best_pose_rot = pose
                        
                # Rule 2: Shoulder shrink fallback
                if baseline_width is not None and shoulder_width is not None:
                    width_ratio = shoulder_width / (baseline_width + 1e-5)
                    if width_ratio < config.SHOULDER_SHRINK_THRESHOLD:
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
