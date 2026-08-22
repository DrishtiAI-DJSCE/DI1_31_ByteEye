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
    
    return max(D_left, D_right) / (D_left + D_right + 1e-5)

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
