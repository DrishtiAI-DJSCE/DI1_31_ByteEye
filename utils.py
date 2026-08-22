def get_keypoint(kpt):
    """
    Safely extracts x, y, and confidence from a keypoint array.
    Supports both (x, y) and (x, y, confidence) formats.
    Returns:
        (x, y, confidence) where confidence is None if not provided.
    """
    if not kpt or len(kpt) < 2:
        return 0.0, 0.0, None
        
    x = float(kpt[0])
    y = float(kpt[1])
    
    if len(kpt) >= 3:
        return x, y, float(kpt[2])
        
    return x, y, None

def calculate_yaw_ratio(center_kpt, left_kpt, right_kpt):
    c_x, c_y, c_conf = center_kpt
    l_x, l_y, l_conf = left_kpt
    r_x, r_y, r_conf = right_kpt
    
    if c_conf is None or l_conf is None or r_conf is None:
        return 0.5
    if c_conf < 0.3:
        return 0.5
        
    D_left = abs(c_x - l_x)
    D_right = abs(c_x - r_x)
    # Signed yaw ratio: ~0.5 is frontal, 0.0 is full right, 1.0 is full left.
    return D_left / (D_left + D_right + 1e-5)

def is_hand_raised(kpts, bbox_height, threshold_ratio):
    if len(kpts) <= 10:
        return False
        
    n_x, n_y, n_c = get_keypoint(kpts[0])
    ls_x, ls_y, ls_c = get_keypoint(kpts[5])
    rs_x, rs_y, rs_c = get_keypoint(kpts[6])
    
    # Scale invariant normalization using shoulder width
    if ls_c is not None and rs_c is not None and ls_c > 0.5 and rs_c > 0.5:
        norm_dist = max(20.0, abs(ls_x - rs_x))
    else:
        norm_dist = max(20.0, bbox_height * 0.3)
        
    # Check left arm
    lw_x, lw_y, lw_c = get_keypoint(kpts[9])
    left_raised = False
    if lw_c is not None and lw_c > 0.6 and n_c is not None and ls_c is not None:
        if lw_y < n_y and lw_y < ls_y:
            if abs(lw_x - ls_x) < threshold_ratio * norm_dist:
                left_raised = True
                
    # Check right arm
    rw_x, rw_y, rw_c = get_keypoint(kpts[10])
    right_raised = False
    if rw_c is not None and rw_c > 0.6 and n_c is not None and rs_c is not None:
        if rw_y < n_y and rw_y < rs_y:
            if abs(rw_x - rs_x) < threshold_ratio * norm_dist:
                right_raised = True
                
    return left_raised or right_raised

def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two bounding boxes (x1, y1, x2, y2)."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-6)
    return iou

def get_bounding_box_containment(inner_box, outer_box):
    """Calculate how much of inner_box is contained within outer_box."""
    x_left = max(inner_box[0], outer_box[0])
    y_top = max(inner_box[1], outer_box[1])
    x_right = min(inner_box[2], outer_box[2])
    y_bottom = min(inner_box[3], outer_box[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    
    return intersection_area / float(inner_area + 1e-6)

import statistics

def robust_stats(values):
    """Return median and Median Absolute Deviation (MAD)."""
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
        
    med = statistics.median(values)
    mad = statistics.median([abs(x - med) for x in values])
    return med, mad
