"""
behaviour.py
============
Temporal behaviour analysis for DRISHTI MVP (CCTV-Optimized & Per-Person State).
"""

import time
import collections
import config
import os
import csv
from severity import calculate_risk_score
from utils import get_keypoint, is_hand_raised, calculate_iou, get_bounding_box_containment, robust_stats

# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class BehaviourAnalyzer:
    def __init__(self):
        self.state = collections.defaultdict(dict)
        self.episode_history = collections.defaultdict(lambda: collections.defaultdict(collections.deque))
        self.track_baselines = {}
        
        # Diagnostic file initialization
        if config.DIAGNOSTIC_MODE:
            self._init_diagnostic_csv()

    def _init_diagnostic_csv(self):
        os.makedirs(os.path.dirname(config.DIAGNOSTIC_CSV_PATH), exist_ok=True)
        # Write header if not exists
        if not os.path.exists(config.DIAGNOSTIC_CSV_PATH):
            with open(config.DIAGNOSTIC_CSV_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "track_id", "nose_shoulder_ratio", "baseline_yaw",
                    "yaw_deviation", "shoulder_width", "baseline_shoulder_width",
                    "shoulder_width_ratio", "wrist_deviation", "wrist_velocity",
                    "phone_confidence", "phone_person_iou", "candidate_anomaly", "notes"
                ])

    def reset(self):
        """Reset all state for a new monitoring session."""
        self.state.clear()
        self.episode_history.clear()
        self.track_baselines.clear()

    def update_track(self, track_id: int, event_type: str, is_occurring: bool,
                     persistence_threshold: float, now_ts: float) -> tuple[bool, bool]:
        """Per-person temporal persistence gate."""
        key = event_type
        
        if key not in self.state[track_id]:
            self.state[track_id][key] = {
                "first_seen": 0,
                "last_seen":  0,
                "active":     False,
                "cooldown_until": 0,
            }

        st = self.state[track_id][key]

        if is_occurring:
            st["last_seen"] = now_ts

        if now_ts < st["cooldown_until"]:
            return False, st["active"]

        newly_confirmed = False

        if is_occurring:
            if st["first_seen"] == 0:
                st["first_seen"] = now_ts

            duration = now_ts - st["first_seen"]
            if duration >= persistence_threshold and not st["active"]:
                st["active"] = True
                st["cooldown_until"] = now_ts + config.EVENT_COOLDOWN_SECONDS
                newly_confirmed = True
        else:
            # Grace period of TRACK_LOSS_GRACE_PERIOD
            if (now_ts - st["last_seen"]) > config.TRACK_LOSS_GRACE_PERIOD:
                if st["first_seen"] != 0:
                    duration = st["last_seen"] - st["first_seen"]
                    self.episode_history[track_id][key].append((st["last_seen"], duration))
                st["first_seen"] = 0
                st["active"] = False

        return newly_confirmed, st["active"]

    def check_frequency_anomaly(self, track_id: int, event_type: str,
                                 window_seconds: float, min_count: int, now_ts: float) -> bool:
        """Detects REPEATED instances for a specific track."""
        history = self.episode_history[track_id][event_type]

        # Remove old episodes
        while history and (now_ts - history[0][0]) > window_seconds:
            history.popleft()

        if len(history) >= min_count:
            history.clear()
            if event_type in self.state[track_id]:
                self.state[track_id][event_type]["cooldown_until"] = now_ts + config.EVENT_COOLDOWN_SECONDS
            return True

        return False

    def process_calibration(self, track_id, raw_yaw: float | None,
                             shoulder_width: float | None,
                             wrist_y: float | None,
                             now_ts: float) -> tuple:
        """Builds stable per-student baselines using robust stats (Median/MAD)."""
        if track_id is None:
            return None, None, None

        if track_id not in self.track_baselines:
            self.track_baselines[track_id] = {
                "start_time":     now_ts,
                "history_yaw":    [],
                "history_width":  [],
                "history_wrist":  [],
                "locked":         False,
                "baseline_yaw":   None,
                "baseline_width": None,
                "baseline_wrist": None,
                "last_seen":      now_ts,
                "smoothing_yaw":  [],
            }

        tb = self.track_baselines[track_id]
        tb["last_seen"] = now_ts

        if tb["locked"]:
            # Rolling wrist baseline (always updates slowly even after locked)
            if wrist_y is not None:
                tb["history_wrist"].append(wrist_y)
                if len(tb["history_wrist"]) > 60: tb["history_wrist"].pop(0)
                med_wrist, _ = robust_stats(tb["history_wrist"])
                tb["baseline_wrist"] = med_wrist
                
            smoothed_nsr = None
            if raw_yaw is not None:
                tb["smoothing_yaw"].append(raw_yaw)
                if len(tb["smoothing_yaw"]) > config.GLANCE_SMOOTHING_FRAMES: tb["smoothing_yaw"].pop(0)
                smoothed_nsr, _ = robust_stats(tb["smoothing_yaw"])

            return tb["baseline_yaw"], tb["baseline_width"], tb["baseline_wrist"], smoothed_nsr

        # Smooth raw yaw before calibration
        smoothed_nsr = None
        if raw_yaw is not None:
            tb["smoothing_yaw"].append(raw_yaw)
            if len(tb["smoothing_yaw"]) > config.GLANCE_SMOOTHING_FRAMES: tb["smoothing_yaw"].pop(0)
            smoothed_nsr, _ = robust_stats(tb["smoothing_yaw"])

        # Only add valid readings
        if smoothed_nsr is not None: tb["history_yaw"].append(smoothed_nsr)
        if shoulder_width is not None: tb["history_width"].append(shoulder_width)
        if wrist_y is not None: tb["history_wrist"].append(wrist_y)

        # Rolling window: keep last ~4s (60 frames)
        if len(tb["history_yaw"]) > 60: tb["history_yaw"].pop(0)
        if len(tb["history_width"]) > 60: tb["history_width"].pop(0)
        if len(tb["history_wrist"]) > 60: tb["history_wrist"].pop(0)

        calibration_elapsed = now_ts - tb["start_time"]

        if calibration_elapsed >= config.CALIBRATION_SECONDS:
            hy = tb["history_yaw"]
            hw = tb["history_width"]

            if len(hy) >= config.CALIBRATION_MIN_FRAMES and len(hw) >= config.CALIBRATION_MIN_FRAMES:
                med_yaw, mad_yaw = robust_stats(hy)
                
                # Check stability using MAD. A stable head posture shouldn't drift more than ~0.05
                if mad_yaw is not None and mad_yaw < 0.05: 
                    tb["baseline_yaw"] = med_yaw
                    med_width, _ = robust_stats(hw)
                    tb["baseline_width"] = med_width
                    med_wrist, _ = robust_stats(tb["history_wrist"])
                    tb["baseline_wrist"] = med_wrist
                    
                    tb["locked"] = True
                    return tb["baseline_yaw"], tb["baseline_width"], tb["baseline_wrist"], smoothed_nsr

            # Not stable yet: slide window forward
            tb["start_time"] = now_ts - (config.CALIBRATION_SECONDS * 0.5)

        med_wrist, _ = robust_stats(tb["history_wrist"]) if tb["history_wrist"] else (None, 0)
        return None, None, med_wrist, smoothed_nsr

    def _nose_shoulder_ratio(self, kpts: list) -> float | None:
        if len(kpts) < 7: return None
        nose_x, nose_y, nose_c = get_keypoint(kpts[0])
        ls_x, ls_y, ls_c = get_keypoint(kpts[5])
        rs_x, rs_y, rs_c = get_keypoint(kpts[6])

        if (nose_c is None or nose_c < config.CCTV_KPT_MIN_CONF or
                ls_c is None or ls_c < config.CCTV_KPT_MIN_CONF or
                rs_c is None or rs_c < config.CCTV_KPT_MIN_CONF):
            return None

        shoulder_mid_x = (ls_x + rs_x) / 2.0
        shoulder_width = abs(ls_x - rs_x)

        if shoulder_width < 8.0: return None
        return abs(nose_x - shoulder_mid_x) / shoulder_width

    def _ear_asymmetry_glance(self, kpts: list) -> bool:
        if len(kpts) < 5: return False
        le_x, le_y, le_c = get_keypoint(kpts[3])
        re_x, re_y, re_c = get_keypoint(kpts[4])
        if le_c is None or re_c is None: return False

        turned_right = (le_c >= config.EAR_VISIBLE_MIN_CONF and re_c <= config.EAR_HIDDEN_MAX_CONF and le_c / (re_c + 1e-6) >= config.EAR_ASYMMETRY_RATIO)
        turned_left = (re_c >= config.EAR_VISIBLE_MIN_CONF and le_c <= config.EAR_HIDDEN_MAX_CONF and re_c / (le_c + 1e-6) >= config.EAR_ASYMMETRY_RATIO)
        return turned_right or turned_left

    def _shoulder_width(self, kpts: list) -> float | None:
        if len(kpts) < 7: return None
        ls_x, ls_y, ls_c = get_keypoint(kpts[5])
        rs_x, rs_y, rs_c = get_keypoint(kpts[6])
        if (ls_c is None or ls_c < config.CCTV_KPT_MIN_CONF or rs_c is None or rs_c < config.CCTV_KPT_MIN_CONF):
            return None
        width = abs(ls_x - rs_x)
        return width if width > 5.0 else None
        
    def _get_avg_wrist_y(self, kpts: list) -> float | None:
        if len(kpts) < 11: return None
        lw_x, lw_y, lw_c = get_keypoint(kpts[9])
        rw_x, rw_y, rw_c = get_keypoint(kpts[10])
        valid = []
        if lw_c is not None and lw_c > config.CCTV_KPT_MIN_CONF: valid.append(lw_y)
        if rw_c is not None and rw_c > config.CCTV_KPT_MIN_CONF: valid.append(rw_y)
        return sum(valid)/len(valid) if valid else None

    def analyze_frame_data(self, persons: list, phones: list, poses: list, frame_dims=None) -> tuple:
        confirmed_events = []
        active_highlights = []
        now_ts = time.time()

        # Clean stale tracks
        stale = [tid for tid, tb in self.track_baselines.items() if (now_ts - tb["last_seen"]) > (config.TRACK_LOSS_GRACE_PERIOD * 2)]
        for tid in stale:
            del self.track_baselines[tid]
            if tid in self.state: del self.state[tid]
            if tid in self.episode_history: del self.episode_history[tid]

        # ----------------------------------------------------------------
        # 1. PHONE / UNAUTHORIZED OBJECT (Associated vs Unassociated)
        # ----------------------------------------------------------------
        associated_phones = {} # track_id -> phone
        unassociated_phones = []
        
        for phone in phones:
            best_iou = 0
            best_track = None
            phone_bbox = phone['bbox']
            
            for pose in poses:
                if 'track_id' not in pose or pose['track_id'] is None: continue
                iou = calculate_iou(phone_bbox, pose['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track = pose['track_id']
                    
            if best_track and best_iou > config.PHONE_PERSON_IOU_THRESHOLD:
                if best_track not in associated_phones or phone['conf'] > associated_phones[best_track]['conf']:
                    associated_phones[best_track] = phone
            else:
                unassociated_phones.append(phone)

        # Handle unassociated phones (Global state)
        if None not in self.state: self.state[None] = {}
        has_unassoc = len(unassociated_phones) > 0
        u_conf, u_act = self.update_track(None, "UNASSOCIATED_PHONE", has_unassoc, config.PHONE_PERSISTENCE_SECONDS, now_ts)
        
        if u_conf:
            best_uphone = max(unassociated_phones, key=lambda p: p['conf'])
            ev = {
                "event_type": "UNASSOCIATED_MOBILE_PHONE",
                "confidence": best_uphone['conf'],
                "bbox": best_uphone['bbox'],
                "timestamp": now_ts,
                "duration": now_ts - self.state[None]["UNASSOCIATED_PHONE"]["first_seen"],
                "frame_dims": frame_dims
            }
            ev["severity"] = calculate_risk_score(ev)
            confirmed_events.append(ev)
            
        if u_act:
            for up in unassociated_phones:
                active_highlights.append({"bbox": up['bbox'], "severity": "HIGH", "label": "UNASSOCIATED PHONE"})

        # ----------------------------------------------------------------
        # 2. POSE-BASED BEHAVIOURS
        # ----------------------------------------------------------------
        for pose in poses:
            track_id = pose.get("track_id")
            if track_id is None: continue
            
            kpts = pose["keypoints"]
            bbox = pose["bbox"]

            nsr = self._nose_shoulder_ratio(kpts)
            ear_glance = self._ear_asymmetry_glance(kpts)
            sw = self._shoulder_width(kpts)
            wrist_y = self._get_avg_wrist_y(kpts)

            baseline_yaw, baseline_width, baseline_wrist, smoothed_nsr = self.process_calibration(track_id, nsr, sw, wrist_y, now_ts)

            # --- Detection Logic ---
            is_turning = False
            yaw_deviation = 0.0
            
            if smoothed_nsr is not None:
                if baseline_yaw is not None:
                    yaw_deviation = abs(smoothed_nsr - baseline_yaw)
                    if yaw_deviation > config.NOSE_SHOULDER_GLANCE_THRESHOLD:
                        is_turning = True

            if not is_turning and ear_glance and smoothed_nsr is not None and baseline_yaw is not None:
                if smoothed_nsr > (baseline_yaw + config.NOSE_SHOULDER_GLANCE_THRESHOLD - 0.05):
                    is_turning = True

            is_rotating = False
            width_ratio = 1.0
            if baseline_width is not None and sw is not None:
                width_ratio = sw / (baseline_width + 1e-6)
                if width_ratio < config.SHOULDER_SHRINK_THRESHOLD:
                    is_rotating = True

            is_hand_up = False
            wrist_deviation = 0.0
            if baseline_wrist is not None and wrist_y is not None:
                # Wrist y is smaller when raised (closer to 0)
                # Significant upward deviation relative to their normal resting position
                # Scale invariant deviation based on bbox height
                bbox_h = bbox[3] - bbox[1]
                wrist_deviation = (baseline_wrist - wrist_y) / (bbox_h + 1e-6)
                if wrist_deviation > 0.25: # Hand raised ~25% of body height above normal resting position
                    is_hand_up = True

            # --- Temporal Gates ---
            turn_conf, turn_act = self.update_track(track_id, "SIDEWARD_GLANCE", is_turning, config.HEAD_TURN_PERSISTENCE_SECONDS, now_ts)
            turn_freq = self.check_frequency_anomaly(track_id, "SIDEWARD_GLANCE", 20.0, 3, now_ts)
            
            rot_conf, rot_act = self.update_track(track_id, "BODY_ROTATION", is_rotating, config.BODY_ROTATION_PERSISTENCE_SECONDS, now_ts)
            rot_freq = self.check_frequency_anomaly(track_id, "BODY_ROTATION", 20.0, 3, now_ts)
            
            hand_conf, hand_act = self.update_track(track_id, "HAND_MOVEMENT", is_hand_up, config.HAND_MOVEMENT_PERSISTENCE_SECONDS, now_ts)
            
            # --- Associated Phone ---
            has_assoc_phone = track_id in associated_phones
            phone_conf, phone_act = self.update_track(track_id, "MOBILE_PHONE", has_assoc_phone, config.PHONE_PERSISTENCE_SECONDS, now_ts)

            # --- Diagnostic Logging ---
            if config.DIAGNOSTIC_MODE:
                candidate = "NORMAL"
                if is_turning or is_rotating or is_hand_up or has_assoc_phone:
                    candidate = "CANDIDATE"
                
                assoc_iou = 0.0
                p_conf = 0.0
                if has_assoc_phone:
                    p_conf = associated_phones[track_id]['conf']
                    assoc_iou = calculate_iou(associated_phones[track_id]['bbox'], bbox)
                
                with open(config.DIAGNOSTIC_CSV_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        now_ts, track_id, smoothed_nsr, baseline_yaw, yaw_deviation, sw, baseline_width, width_ratio,
                        wrist_deviation, 0.0, p_conf, assoc_iou, candidate, ""
                    ])

            # --- Emitting Events ---
            if turn_conf or turn_freq:
                fs = self.state[track_id]["SIDEWARD_GLANCE"]["first_seen"]
                ev = {"event_type": "SIDEWARD_GLANCE", "confidence": 1.0, "bbox": bbox, "timestamp": now_ts,
                      "duration": now_ts - fs if fs > 0 else 0.0, "frame_dims": frame_dims}
                ev["severity"] = calculate_risk_score(ev)
                confirmed_events.append(ev)
            if turn_act: active_highlights.append({"bbox": bbox, "severity": "MEDIUM", "label": "SIDEWARD GLANCE"})

            if rot_conf or rot_freq:
                fs = self.state[track_id]["BODY_ROTATION"]["first_seen"]
                ev = {"event_type": "BODY_ROTATION", "confidence": 1.0, "bbox": bbox, "timestamp": now_ts,
                      "duration": now_ts - fs if fs > 0 else 0.0, "frame_dims": frame_dims}
                ev["severity"] = calculate_risk_score(ev)
                confirmed_events.append(ev)
            if rot_act: active_highlights.append({"bbox": bbox, "severity": "MEDIUM", "label": "BODY ROTATION"})
            
            if hand_conf:
                ev = {"event_type": "HAND_MOVEMENT", "confidence": 1.0, "bbox": bbox, "timestamp": now_ts,
                      "duration": now_ts - self.state[track_id]["HAND_MOVEMENT"]["first_seen"], "frame_dims": frame_dims}
                ev["severity"] = calculate_risk_score(ev)
                confirmed_events.append(ev)
            if hand_act: active_highlights.append({"bbox": bbox, "severity": "MEDIUM", "label": "HAND MOVEMENT"})
            
            if phone_conf:
                ev = {"event_type": "MOBILE_PHONE", "confidence": associated_phones[track_id]['conf'], "bbox": associated_phones[track_id]['bbox'], "timestamp": now_ts,
                      "duration": now_ts - self.state[track_id]["MOBILE_PHONE"]["first_seen"], "frame_dims": frame_dims}
                ev["severity"] = calculate_risk_score(ev)
                confirmed_events.append(ev)
            if phone_act: 
                p_bbox = associated_phones[track_id]['bbox'] if track_id in associated_phones else bbox
                active_highlights.append({"bbox": p_bbox, "severity": "HIGH", "label": "MOBILE PHONE"})

        return confirmed_events, active_highlights
