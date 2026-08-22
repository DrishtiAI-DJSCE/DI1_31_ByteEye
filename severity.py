import config
import datetime

# Object-detected event types where confidence comes from the vision model
# and is a meaningful discriminator (vs. behavior events where confidence
# is always 1.0 from the behavior confirmation system).
_OBJECT_DETECTED_EVENTS = {"MOBILE_PHONE", "UNASSOCIATED_MOBILE_PHONE"}

def calculate_risk_score(event_data: dict) -> str:
    """
    Calculates a multi-factor risk score for an event and returns its severity string.

    The scoring system uses a multi-factor approach where:
    - Behavior type sets the base risk level (from config weights)
    - Duration/persistence provides continuous temporal escalation
    - Detection confidence matters for object-detected events (e.g. phone)
    - Bounding box size is a minor supporting factor (not primary)
    - Frequency anomalies (repeated episodes) significantly escalate risk

    Designed to work correctly with wide-angle CCTV footage where objects
    appear smaller in frame — bounding box size does NOT dominate the score.

    event_data must contain:
      - event_type (str)
    event_data may contain:
      - duration (float)
      - confidence (float)
      - bbox (list)
      - frame_dims (tuple: (h, w, c))
      - is_frequency_anomaly (bool)
    """
    event_type = event_data.get("event_type", "UNKNOWN")
    duration = event_data.get("duration", 0.0)
    confidence = event_data.get("confidence", 0.5)
    bbox = event_data.get("bbox")
    frame_dims = event_data.get("frame_dims")
    is_frequency_anomaly = event_data.get("is_frequency_anomaly", False)

    score = 0

    # -------------------------------------------------------------------
    # 1. Behavior Base Score (primary factor)
    # The inherent risk level of this behavior type, from config.
    # Range: typically 10–60, directly sets the risk baseline.
    # -------------------------------------------------------------------
    behavior_base = config.BEHAVIOR_WEIGHTS.get(event_type, 10)
    score += behavior_base

    # -------------------------------------------------------------------
    # 2. Temporal Persistence / Duration Score
    # How long the behavior has been continuously active.
    # Continuous scaling: +3 points per second of duration, capped at +30.
    # This replaces the old step function (+5/+10/+20) which always hit
    # the same step because events fire at exactly the persistence
    # threshold.  Continuous scaling allows prolonged behaviors to
    # escalate meaningfully across risk bands.
    #
    # Examples at various durations:
    #   0.5s → +1.5    (phone just confirmed)
    #   3.0s → +9.0    (behavior just confirmed at persistence threshold)
    #   5.0s → +15.0   (sustained behavior)
    #   8.0s → +24.0   (prolonged behavior)
    #  10.0s → +30.0   (capped — maximum temporal escalation)
    # -------------------------------------------------------------------
    duration_bonus = min(30.0, duration * 3.0)
    score += duration_bonus

    # -------------------------------------------------------------------
    # 3. Detection Confidence Score
    #
    # For OBJECT-DETECTED events (e.g. MOBILE_PHONE), the confidence
    # value comes from the vision model and varies meaningfully (0.4–1.0).
    # Higher confidence → model is more certain → higher risk contribution.
    #
    # For BEHAVIOR events (SIDEWARD_GLANCE, BODY_ROTATION, HAND_MOVEMENT),
    # confidence is always hardcoded to 1.0 in behaviour.py.  It signals
    # "the behavior system confirmed this is occurring" — a binary flag,
    # NOT a varying signal.  Adding a flat +5 bonus to EVERY behavior
    # event (as the old code did) just shifted ALL scores by a constant,
    # providing zero discrimination.  These events are already validated
    # by temporal persistence, so no confidence bonus is needed.
    # -------------------------------------------------------------------
    confidence_bonus = 0
    if event_type in _OBJECT_DETECTED_EVENTS:
        # Phone/object detections: confidence is the actual detector score
        if confidence >= 0.8:
            confidence_bonus = 10
        elif confidence >= 0.6:
            confidence_bonus = 5
        elif confidence < 0.4:
            confidence_bonus = -10
    score += confidence_bonus

    # -------------------------------------------------------------------
    # 4. Spatial Context / Size Score  (minor supporting factor)
    #
    # Bounding-box size relative to the frame area.  This is a
    # SUPPORTING signal only — it must NOT dominate risk.
    #
    # Wide-angle cameras naturally produce smaller bounding boxes.
    # A person far from the camera is NOT less suspicious.  The size
    # factor is capped to ±5 points and has a generous neutral zone
    # so that typical wide-angle detections receive 0 penalty.
    # -------------------------------------------------------------------
    size_bonus = 0
    norm_size = 0.0
    if bbox and frame_dims:
        h, w = frame_dims[:2]
        frame_area = h * w
        if frame_area > 0:
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]
            bbox_area = box_w * box_h
            norm_size = bbox_area / frame_area

            # Wide-angle perspective correction: objects are artificially
            # smaller, so boost the normalized size for fairer comparison.
            if getattr(config, "WIDE_ANGLE_MODE", False):
                norm_size = min(norm_size * 2.5, 0.5)

            # Scoring: generous neutral zone, minimal penalty for small
            if norm_size > 0.15:
                size_bonus = 5      # large / close-up object
            elif norm_size > 0.03:
                size_bonus = 2      # moderate size
            elif norm_size >= 0.005:
                size_bonus = 0      # neutral — normal wide-angle range
            else:
                size_bonus = -2     # very tiny — slight reduction only
    score += size_bonus

    # -------------------------------------------------------------------
    # 5. Frequency Anomaly Score
    #
    # When the behavior system detects repeated episodes of the same
    # behavior within a short time window (e.g. 3 sideward glances in
    # 15 seconds), it sets is_frequency_anomaly=True.
    #
    # This is a significant escalation (+25 points) that can push an
    # otherwise MEDIUM event into HIGH, reflecting that repeated
    # suspicious activity is more concerning than a single occurrence.
    # -------------------------------------------------------------------
    frequency_bonus = 25 if is_frequency_anomaly else 0
    score += frequency_bonus

    # -------------------------------------------------------------------
    # Final classification
    # -------------------------------------------------------------------
    final_score = max(0, min(100, int(score)))

    if event_type in {"MOBILE_PHONE", "UNASSOCIATED_MOBILE_PHONE"}:
        classification = "HIGH"
    elif final_score <= config.RISK_THRESHOLDS["LOW_MAX"]:
        classification = "LOW"
    elif final_score <= config.RISK_THRESHOLDS["MEDIUM_MAX"]:
        classification = "MEDIUM"
    else:
        classification = "HIGH"

    # Logging
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Risk Classification Debug:")
    print(f"  Event: {event_type}")
    print(f"  Duration: {duration:.2f}s | Confidence: {confidence:.2f} | Norm Size: {norm_size:.4f}")
    print(f"  Breakdown: behavior={behavior_base} + duration={duration_bonus:.1f}"
          f" + confidence={confidence_bonus} + size={size_bonus}"
          f" + frequency={frequency_bonus}")
    print(f"  Raw Score: {score:.1f} -> Final Score: {final_score}")
    print(f"  Result: {classification}")

    return classification

def get_severity(event_type: str) -> str:
    """Legacy wrapper for simple string-based lookups if any still exist."""
    return calculate_risk_score({"event_type": event_type})
