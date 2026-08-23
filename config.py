"""
config.py
=========
Single source of truth for all runtime settings.

Uses python-dotenv to load .env from the project root, so every value
here can be overridden without touching code. Copy .env.example to .env
and change whatever you need — the application picks it up automatically
on the next run.

IMPORTANT: This module calls load_dotenv() at import time.
Import it before any other project module.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the directory that contains this file (project root).
# If .env doesn't exist, dotenv silently does nothing — the defaults below
# remain in effect, which is the correct behaviour for a fresh install.
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)


def _bool(key: str, default: bool) -> bool:
    """Parse an environment variable as a boolean."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _float(key: str, default: float) -> float:
    """Parse an environment variable as a float."""
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _int(key: str, default: int) -> int:
    """Parse an environment variable as an int."""
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Model paths — point these at a fine-tuned model once you have one
# ---------------------------------------------------------------------------
OBJECT_MODEL_PATH: str = os.getenv("OBJECT_MODEL_PATH", "models/drishti_detector.pt")
POSE_MODEL_PATH: str = os.getenv("POSE_MODEL_PATH", "models/yolov8n-pose.pt")
DEVICE: str = os.getenv("DEVICE", "auto")  # "auto" | "cpu" | "cuda" | "mps"

# ---------------------------------------------------------------------------
# Inference & processing
# ---------------------------------------------------------------------------
TARGET_FPS: int = _int("ANALYSIS_FPS", 10) # Deprecated, use AI_TARGET_FPS
AI_TARGET_FPS: int = _int("AI_TARGET_FPS", 10)  # 8–15 is the realistic range
WIDE_ANGLE_MODE: bool = _bool("WIDE_ANGLE_MODE", True)
INFERENCE_IMGSZ: int = _int("INFERENCE_IMGSZ", 960 if WIDE_ANGLE_MODE else 640)

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------
PERSON_CONFIDENCE: float = _float("OBJECT_CONFIDENCE_THRESHOLD", 0.25)
PHONE_CONFIDENCE: float = _float("PHONE_CONFIDENCE_THRESHOLD", 0.40)
POSE_CONFIDENCE: float = _float("POSE_CONFIDENCE_THRESHOLD", 0.50)

# ---------------------------------------------------------------------------
# Behaviour thresholds (geometric ratios / pixel values)
# ---------------------------------------------------------------------------
HEAD_TURN_THRESHOLD_RATIO: float = _float("HEAD_TURN_THRESHOLD", 0.85)
BODY_ROTATION_THRESHOLD_RATIO: float = _float("BODY_ROTATION_THRESHOLD", 0.85)
HAND_MOVEMENT_THRESHOLD: float = _float("HAND_MOVEMENT_THRESHOLD", 0.5)

# Per-Seat Baseline Calibration
CALIBRATION_SECONDS: float = _float("CALIBRATION_SECONDS", 5.0)
YAW_DEVIATION_THRESHOLD: float = _float("YAW_DEVIATION_THRESHOLD", 0.20)
SHOULDER_SHRINK_THRESHOLD: float = _float("SHOULDER_SHRINK_THRESHOLD", 0.80)

# ---------------------------------------------------------------------------
# Temporal persistence — how long a behaviour must persist before it is
# considered a confirmed anomaly.  Prevents single-frame false alarms.
# ---------------------------------------------------------------------------
HEAD_TURN_PERSISTENCE_SECONDS: float = _float("HEAD_TURN_PERSISTENCE_SECONDS", 3.0)
BODY_ROTATION_PERSISTENCE_SECONDS: float = _float("BODY_ROTATION_PERSISTENCE_SECONDS", 3.0)
HAND_MOVEMENT_PERSISTENCE_SECONDS: float = _float("HAND_MOVEMENT_PERSISTENCE_SECONDS", 3.0)
PHONE_PERSISTENCE_SECONDS: float = _float("PHONE_PERSISTENCE_SECONDS", 0.5)

# ---------------------------------------------------------------------------
# Event cooldown — after an event fires, ignore the same category for
# this many seconds to prevent duplicate database rows / snapshots.
# ---------------------------------------------------------------------------
EVENT_COOLDOWN_SECONDS: float = _float("EVENT_COOLDOWN_SECONDS", 10.0)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DB_PATH: str = os.getenv("DATABASE_PATH", "database/events.db")
EVIDENCE_DIR: str = os.getenv("EVIDENCE_DIR", "evidence")
DELETE_SESSION_EVIDENCE_ON_STOP: bool = _bool("DELETE_SESSION_EVIDENCE_ON_STOP", True)

# ---------------------------------------------------------------------------
# Debug & Diagnostics
# ---------------------------------------------------------------------------
DEBUG_MODE: bool = _bool("DEBUG_MODE", False)

DIAGNOSTIC_MODE: bool = _bool("DIAGNOSTIC_MODE", False)
DIAGNOSTIC_CSV_PATH: str = os.getenv("DIAGNOSTIC_CSV_PATH", "data/diagnostics.csv")

# ---------------------------------------------------------------------------
# CCTV Behaviour Geometry Constants
# ---------------------------------------------------------------------------
CCTV_KPT_MIN_CONF: float = _float("CCTV_KPT_MIN_CONF", 0.35)
EAR_VISIBLE_MIN_CONF: float = _float("EAR_VISIBLE_MIN_CONF", 0.40)
EAR_HIDDEN_MAX_CONF: float = _float("EAR_HIDDEN_MAX_CONF", 0.28)
EAR_ASYMMETRY_RATIO: float = _float("EAR_ASYMMETRY_RATIO", 2.0)
NOSE_SHOULDER_GLANCE_THRESHOLD: float = _float("NOSE_SHOULDER_GLANCE_THRESHOLD", 0.15)
GLANCE_SMOOTHING_FRAMES: int = _int("GLANCE_SMOOTHING_FRAMES", 7)

# Phone association IoU
PHONE_PERSON_IOU_THRESHOLD: float = _float("PHONE_PERSON_IOU_THRESHOLD", 0.1)

# Rolling history settings
CALIBRATION_MIN_FRAMES: int = _int("CALIBRATION_MIN_FRAMES", 15)
TRACK_LOSS_GRACE_PERIOD: float = _float("TRACK_LOSS_GRACE_PERIOD", 1.0)

# ---------------------------------------------------------------------------
# Risk Classification Thresholds & Weights
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "LOW_MAX": _int("RISK_LOW_MAX", 30),
    "MEDIUM_MAX": _int("RISK_MEDIUM_MAX", 65)
}

BEHAVIOR_WEIGHTS = {
    "MOBILE_PHONE": _int("WEIGHT_MOBILE_PHONE", 60),
    "BODY_ROTATION": _int("WEIGHT_BODY_ROTATION", 40),
    "SIDEWARD_GLANCE": _int("WEIGHT_SIDEWARD_GLANCE", 35),
    "HAND_MOVEMENT": _int("WEIGHT_HAND_MOVEMENT", 20)
}
