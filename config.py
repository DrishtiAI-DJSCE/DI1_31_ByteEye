import os

# Model Paths
OBJECT_MODEL_PATH = os.getenv("OBJECT_MODEL_PATH", "models/yolov8n.pt")
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "models/yolov8n-pose.pt")

# Inference & Processing
TARGET_FPS = 10  # approximate AI FPS (target 8-15)

# Confidence Thresholds
PERSON_CONFIDENCE = 0.5
PHONE_CONFIDENCE = 0.5
POSE_CONFIDENCE = 0.5

# Behaviour Thresholds
# We'll refine these metrics in the behaviour module
HEAD_TURN_THRESHOLD_RATIO = 0.25 # Ratio based distance metric
BODY_ROTATION_THRESHOLD_RATIO = 0.25
HAND_MOVEMENT_THRESHOLD = 0.5

# Temporal Persistence (Seconds)
HEAD_TURN_PERSISTENCE_SECONDS = 3.0
BODY_ROTATION_PERSISTENCE_SECONDS = 3.0
HAND_MOVEMENT_PERSISTENCE_SECONDS = 3.0
PHONE_PERSISTENCE_SECONDS = 2.0

# Event Cooldown
EVENT_COOLDOWN_SECONDS = 30.0

# Storage
DB_PATH = "drishti_events.db"
EVIDENCE_DIR = "evidence"

# Debug
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
