import streamlit as st
import cv2
import time
import os
import datetime
import tempfile
from PIL import Image

import config
from vision import VisionAnalyzer
from behaviour import BehaviourAnalyzer
from severity import get_severity
import evidence
import database

st.set_page_config(page_title="DRISHTI MVP", layout="wide")
database.init_db()

if 'setup_done' not in st.session_state: st.session_state['setup_done'] = False
if 'monitoring' not in st.session_state: st.session_state['monitoring'] = False
if 'behaviour_analyzer' not in st.session_state: st.session_state['behaviour_analyzer'] = BehaviourAnalyzer()
if 'vision' not in st.session_state: st.session_state['vision'] = None
if 'stop_monitoring' not in st.session_state: st.session_state['stop_monitoring'] = False
if 'video_path' not in st.session_state: st.session_state['video_path'] = None

def setup_page():
    st.title("DRISHTI")
    st.subheader("AI Examination Monitoring System")
    st.text_input("Exam Name")
    st.text_input("Hall / Room")
    
    mode = st.radio("Monitoring Mode", ["Live Camera", "Uploaded Video"])
    
    if mode == "Uploaded Video":
        uploaded_file = st.file_uploader("Upload Exam Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            st.write(f"✓ Video loaded: {uploaded_file.name}")
            if st.session_state['video_path'] is None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                st.session_state['video_path'] = tfile.name
                
            if st.button("Start Monitoring"):
                st.session_state['setup_done'] = True
                st.session_state['monitoring'] = True
                st.session_state['stop_monitoring'] = False
                st.rerun()
        else:
            st.write("Please upload an exam video.")
    else:
        st.session_state['video_path'] = 0 # Camera ID 0
        if st.button("Start Monitoring"):
            st.session_state['setup_done'] = True
            st.session_state['monitoring'] = True
            st.session_state['stop_monitoring'] = False
            st.rerun()

def get_color(severity):
    if severity == "HIGH":
        return (255, 0, 0) # Red in RGB, but OpenCV uses BGR natively, wait Streamlit shows RGB. We will draw BGR then convert.
        # Let's return BGR colors since cv2 draws in BGR, and then we cvtColor
    if severity == "HIGH": return (0, 0, 255) # BGR Red
    if severity == "MEDIUM": return (0, 165, 255) # BGR Orange
    return (0, 255, 0) # BGR Green

def monitoring_page():
    if st.session_state['vision'] is None:
        with st.spinner("Loading AI Models..."):
            st.session_state['vision'] = VisionAnalyzer()
        
    st.title("DRISHTI - LIVE MONITORING")
    
    mode_str = "LIVE CAMERA" if st.session_state['video_path'] == 0 else "UPLOADED VIDEO"
    st.write(f"Status: 🟢 LIVE | Mode: {mode_str} | Target FPS: {config.TARGET_FPS}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    students_detected = st.empty()
    active_alerts = st.empty()
    high_risk = st.empty()
    med_risk = st.empty()
    low_risk = st.empty()
    
    main_col, alert_col = st.columns([3, 1])
    
    with main_col:
        video_placeholder = st.empty()
        
    with alert_col:
        st.subheader("LIVE ALERTS")
        alerts_placeholder = st.empty()
        
    if st.sidebar.button("Stop Monitoring"):
        st.session_state['stop_monitoring'] = True
        st.session_state['monitoring'] = False
        st.rerun()
        
    cap = cv2.VideoCapture(st.session_state['video_path'])
    
    while not st.session_state['stop_monitoring']:
        ret, frame = cap.read()
        if not ret:
            if st.session_state['video_path'] != 0:
                st.info("Video playback finished.")
            else:
                st.error("Camera disconnected.")
            break
            
        t0 = time.time()
        
        # Analyze
        res = st.session_state['vision'].process_frame(frame)
        
        # Behaviors
        confirmed_anomalies, active_highlights = st.session_state['behaviour_analyzer'].analyze_frame_data(
            res['persons'], res['phones'], res['poses']
        )
        
        # Save anomalies
        for anomaly in confirmed_anomalies:
            snapshot_path = evidence.save_evidence(frame, anomaly)
            anomaly['snapshot_path'] = snapshot_path
            database.insert_event(anomaly)
            
        # Draw Live View
        viz_frame = frame.copy()
        
        # 1. CONTINUOUS: Default boxes for ALL persons
        for person in res['persons']:
            x1, y1, x2, y2 = map(int, person['bbox'])
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (255, 0, 0), 1) # Blue BGR for normal students
            
        # 2. CONTINUOUS: Default skeletons for ALL poses
        for pose in res['poses']:
            for kpt in pose['keypoints']:
                x, y, conf = kpt
                if conf > 0.5:
                    cv2.circle(viz_frame, (int(x), int(y)), 3, (0, 255, 255), -1) # Yellow BGR
                    
        # 3. CONTINUOUS: Default boxes for ALL phones
        for phone in res['phones']:
            x1, y1, x2, y2 = map(int, phone['bbox'])
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (255, 255, 0), 1) # Cyan BGR
            
        # 4. EVENT-DRIVEN: Active Highlights
        for highlight in active_highlights:
            bbox = highlight['bbox']
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                color = get_color(highlight['severity'])
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 3) # Thicker box
                
                # Label
                label = f"{highlight['severity']} - {highlight['label']}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(viz_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(viz_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        viz_frame = cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(viz_frame, channels="RGB", use_container_width=True)
        
        # Update KPIs
        events = database.get_events()
        h_c = sum(1 for e in events if e['severity'] == 'HIGH' and e['status'] == 'NEW')
        m_c = sum(1 for e in events if e['severity'] == 'MEDIUM' and e['status'] == 'NEW')
        l_c = sum(1 for e in events if e['severity'] == 'LOW' and e['status'] == 'NEW')
        
        students_detected.metric("Students Detected", len(res['persons']))
        active_alerts.metric("Active Alerts", h_c + m_c + l_c)
        high_risk.metric("High Risk", h_c)
        med_risk.metric("Medium Risk", m_c)
        low_risk.metric("Low Risk", l_c)
        
        # Update Alerts Sidebar
        new_events = [e for e in events if e['status'] == 'NEW'][:5]
        with alerts_placeholder.container():
            if not new_events:
                st.write("No new alerts.")
            for e in new_events:
                ts_str = datetime.datetime.fromtimestamp(e['timestamp']).strftime("%H:%M:%S")
                color = "🔴" if e['severity'] == "HIGH" else "🟠" if e['severity'] == "MEDIUM" else "🟢"
                st.markdown(f"**{color} {e['severity']}**")
                st.markdown(f"{e['event_type'].replace('_', ' ').title()}")
                st.markdown(f"{ts_str}")
                st.markdown("---")
                
        # Simple FPS target
        elapsed = time.time() - t0
        sleep_time = max(0, (1.0 / config.TARGET_FPS) - elapsed)
        time.sleep(sleep_time)

    cap.release()

def event_history_page():
    st.title("Event History")
    events = database.get_events()
    
    if not events:
        st.write("No events recorded.")
        return
        
    for e in events:
        ts_str = datetime.datetime.fromtimestamp(e['timestamp']).strftime("%H:%M:%S")
        with st.expander(f"{ts_str} | {e['severity']} | {e['event_type']} | {e['status']}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**Severity:** {e['severity']}")
                st.write(f"**Confidence:** {e['confidence']:.2f}")
                
                if e['status'] == 'NEW':
                    if st.button("Mark Reviewed", key=f"rev_{e['event_id']}"):
                        database.update_event_status(e['event_id'], "REVIEWED")
                        st.rerun()
                    if st.button("Dismiss", key=f"dis_{e['event_id']}"):
                        database.update_event_status(e['event_id'], "DISMISSED")
                        st.rerun()
            with col2:
                if e['snapshot_path'] and os.path.exists(e['snapshot_path']):
                    st.image(e['snapshot_path'], use_container_width=True)

def main():
    st.sidebar.title("Navigation")
    
    if not st.session_state['setup_done']:
        setup_page()
    else:
        page = st.sidebar.radio("Go to", ["Live Monitoring", "Event History"])
        if page == "Live Monitoring":
            monitoring_page()
        else:
            event_history_page()
            
    with st.sidebar.expander("Advanced / Developer Settings"):
        st.write(f"Object Confidence: {config.PERSON_CONFIDENCE}")
        st.write(f"Phone Confidence: {config.PHONE_CONFIDENCE}")
        st.write(f"Analysis FPS Target: {config.TARGET_FPS}")
        st.write(f"Cooldown (s): {config.EVENT_COOLDOWN_SECONDS}")

if __name__ == "__main__":
    main()
