import os
import cv2
import datetime
import config

def save_evidence(frame, event):
    severity = event.get('severity', 'LOW').lower()
    ts_dt = datetime.datetime.fromtimestamp(event['timestamp'])
    ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
    file_ts_str = ts_dt.strftime("%Y-%m-%d_%H-%M-%S")
    
    event_type = event.get('event_type', 'unknown').lower()
    
    filename = f"{file_ts_str}_{event_type}.jpg"
    
    out_dir = os.path.join(config.EVIDENCE_DIR, severity)
    os.makedirs(out_dir, exist_ok=True)
        
    out_path = os.path.join(out_dir, filename)
    
    viz_frame = frame.copy()
    
    if 'bbox' in event and event['bbox']:
        x1, y1, x2, y2 = map(int, event['bbox'])
        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    overlay_lines = [
        "DRISHTI",
        f"{severity.upper()} - {event_type.replace('_', ' ').upper()}",
        ts_str
    ]
    if 'confidence' in event:
        overlay_lines.append(f"Confidence: {event['confidence']:.2f}")
        
    y_offset = 30
    for line in overlay_lines:
        (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(viz_frame, (20, y_offset - text_height - 5), (20 + text_width, y_offset + 5), (0, 0, 0), -1)
        cv2.putText(viz_frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 35
        
    cv2.imwrite(out_path, viz_frame)
    return out_path
