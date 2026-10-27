import cv2
import numpy as np
import math

# Network input is a 192x192 square.
POSE_INP_SIZE = 192

# Vertical shift of the ROI center below the shoulder line, in pixels.
ROI_LOWER_OFFSET = 35

# ROI size as a fraction of shoulder width and trunk length.
ROI_W_MULT = 0.30
ROI_H_MULT = 2.94

# Aspect-ratio limits that keep the analysis window from collapsing or stretching.
MIN_ASPECT_RATIO = 0.04
MAX_ASPECT_RATIO = 1.20
_net = None
_net_checked = False


def get_net():
    """Load the pose network on first use. Missing weights leave ROI detection off."""
    global _net, _net_checked
    if _net_checked:
        return _net
    _net_checked = True
    from acquisition.settings import pose_onnx_path

    path = pose_onnx_path()
    if not path.is_file():
        print(f"Pose network is missing. Place the ONNX file at: {path}")
        return None
    print(f"Loading pose network: {path}")
    try:
        _net = cv2.dnn.readNetFromONNX(str(path))
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception as exc:
        print(f"Could not load the pose network ({path}): {exc}")
        _net = None
    return _net

def get_pose_roi(frame):
    """Return the chest ROI and keypoints, or (None, None) when the network is unavailable."""
    net = get_net()
    if net is None:
        return None, None

    h, w = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (POSE_INP_SIZE, POSE_INP_SIZE), (128, 128, 128), False, False)
    net.setInput(blob)
    out = net.forward()
    out = out[0] 

    keypoints = []
    _, Hm, Wm = out.shape 

    # Each of the 18 heatmaps votes for one joint.
    for i in range(18): 
        if i >= out.shape[0]: 
            keypoints.append(None)
            continue 
            
        heatmap = out[i, :, :] 
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        # Joints below this confidence are dropped.
        if conf > 0.1:
            x = (point[0] / (Wm - 1)) * w
            y = (point[1] / (Hm - 1)) * h
            keypoints.append((int(x), int(y)))
        else:
            keypoints.append(None)

    # Midpoints of the right and left shoulder segments, measured from the neck.
    mid_right = None
    if keypoints[1] is not None and keypoints[2] is not None:
        mx = int((keypoints[1][0] + keypoints[2][0]) / 2)
        my = int((keypoints[1][1] + keypoints[2][1]) / 2)
        mid_right = (mx, my)
    keypoints.append(mid_right) 

    mid_left = None
    if keypoints[1] is not None and keypoints[5] is not None:
        mx = int((keypoints[1][0] + keypoints[5][0]) / 2)
        my = int((keypoints[1][1] + keypoints[5][1]) / 2)
        mid_left = (mx, my)
    keypoints.append(mid_left) 
    
    # Chest center, then the same point shifted down onto the thorax.
    center_y_point = None
    lower_center_point = None
    if keypoints[1] is not None:
        valid_ys = []
        for idx in [1, 2, 5, 18, 19]:
            if idx < len(keypoints) and keypoints[idx] is not None:
                valid_ys.append(keypoints[idx][1])
                
        if len(valid_ys) > 0:
            avg_y = int(sum(valid_ys) / len(valid_ys))
            center_y_point = (keypoints[1][0], avg_y)
            lower_center_point = (keypoints[1][0], avg_y + ROI_LOWER_OFFSET)
            
    keypoints.append(center_y_point) 
    keypoints.append(lower_center_point) 
    
    roi_coords = None
    if lower_center_point is not None and keypoints[1] is not None and keypoints[2] is not None and keypoints[5] is not None:
        cx, cy = lower_center_point
        
        # Shoulder span is the reference width.
        dx_shoulders = keypoints[2][0] - keypoints[5][0]
        dy_shoulders = keypoints[2][1] - keypoints[5][1]
        raw_w = math.sqrt(dx_shoulders**2 + dy_shoulders**2)
        
        # Trunk length is the reference height.
        dx_trunk = keypoints[1][0] - cx
        dy_trunk = keypoints[1][1] - cy
        raw_h = math.sqrt(dx_trunk**2 + dy_trunk**2)
        
        target_w = raw_w * ROI_W_MULT
        target_h = raw_h * ROI_H_MULT
        
        if target_h > 0:
            current_ratio = target_w / target_h
            if current_ratio < MIN_ASPECT_RATIO:
                target_w = target_h * MIN_ASPECT_RATIO
            elif current_ratio > MAX_ASPECT_RATIO:
                target_h = target_w / MAX_ASPECT_RATIO
                
        target_w = int(target_w)
        target_h = int(target_h)
        
        x1 = int(cx - target_w / 2)
        y1 = int(cy - target_h / 2)
        
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        
        roi_w_safe = min(target_w, w - x1)
        roi_h_safe = min(target_h, h - y1)
        
        if roi_w_safe >= 10 and roi_h_safe >= 10:
            roi_coords = [x1, y1, roi_w_safe, roi_h_safe]
            
    return roi_coords, keypoints