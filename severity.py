def get_severity(event_type):
    """
    Returns HIGH, MEDIUM, or LOW based on event type.
    """
    if event_type == "MOBILE_PHONE":
        return "HIGH"
    elif event_type in ["SIDEWARD_GLANCE", "BODY_ROTATION", "HAND_MOVEMENT"]:
        return "MEDIUM"
    else:
        return "LOW"
