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

# Initialize session state variables
if 'setup_done' not in st.session_state: st.session_state['setup_done'] = False
if 'monitoring' not in st.session_state: st.session_state['monitoring'] = False
if 'behaviour_analyzer' not in st.session_state: st.session_state['behaviour_analyzer'] = BehaviourAnalyzer()
if 'vision' not in st.session_state: st.session_state['vision'] = None
if 'stop_monitoring' not in st.session_state: st.session_state['stop_monitoring'] = False
if 'input_mode' not in st.session_state: st.session_state['input_mode'] = None
if 'uploaded_video_path' not in st.session_state: st.session_state['uploaded_video_path'] = None
if 'camera_open' not in st.session_state: st.session_state['camera_open'] = False
if 'video_open' not in st.session_state: st.session_state['video_open'] = False
if 'frame_idx' not in st.session_state: st.session_state['frame_idx'] = 0
if 'session_id' not in st.session_state: st.session_state['session_id'] = None

def cleanup_session():
    if 'session_id' in st.session_state and st.session_state['session_id']:
        sid = st.session_state['session_id']
        if config.DELETE_SESSION_EVIDENCE_ON_STOP:
            database.delete_session_events(sid)
            evidence.clear_session_evidence(sid)
    
    if st.session_state.get('uploaded_video_path') and os.path.exists(st.session_state['uploaded_video_path']):
        try:
            os.remove(st.session_state['uploaded_video_path'])
        except Exception:
            pass
            
    st.session_state['behaviour_analyzer'].reset()
    st.session_state['uploaded_video_path'] = None
    st.session_state['frame_idx'] = 0
    st.session_state['session_id'] = None
    st.session_state['stop_monitoring'] = True
    st.session_state['monitoring'] = False
    st.session_state['setup_done'] = False


def setup_page():
    st.title("DRISHTI")
    st.subheader("AI Examination Monitoring System")
    st.text_input("Exam Name")
    st.text_input("Hall / Room")
    
    mode = st.radio("Monitoring Mode", ["Live Camera", "Uploaded Video"])
    
    if mode == "Uploaded Video":
        st.session_state['input_mode'] = "UPLOADED_VIDEO"
        uploaded_file = st.file_uploader("Upload Exam Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            st.write(f"✓ Video loaded: {uploaded_file.name}")
            
            if st.session_state['uploaded_video_path'] is None or not os.path.exists(st.session_state['uploaded_video_path']):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.close()
                st.session_state['uploaded_video_path'] = tfile.name
                st.session_state['frame_idx'] = 0 # reset on new upload
                
            if st.button("Start Monitoring"):
                st.session_state['session_id'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                st.session_state['setup_done'] = True
                st.session_state['monitoring'] = True
                st.session_state['stop_monitoring'] = False
                st.rerun()
        else:
            st.write("Please upload an exam video.")
            st.session_state['uploaded_video_path'] = None
            st.session_state['frame_idx'] = 0
    else:
        st.session_state['input_mode'] = "LIVE_CAMERA"
        st.write("Camera Source")
        st.write("[default webcam]")
        if st.button("Start Monitoring"):
            st.session_state['session_id'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            st.session_state['setup_done'] = True
            st.session_state['monitoring'] = True
            st.session_state['stop_monitoring'] = False
            st.rerun()

def get_color(severity):
    if severity == "HIGH": return (0, 0, 255) # BGR Red
    if severity == "MEDIUM": return (0, 165, 255) # BGR Orange
    return (0, 255, 0) # BGR Green

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def monitoring_page():
    if st.session_state['vision'] is None:
        with st.spinner("Loading AI Models..."):
            st.session_state['vision'] = VisionAnalyzer()
        
    # --- Top Header ---
    mode_str = "LIVE CAMERA" if st.session_state['input_mode'] == "LIVE_CAMERA" else "UPLOADED VIDEO"
    st.markdown(f"### DRISHTI &nbsp;&nbsp;&nbsp; 🟢 LIVE &nbsp;&nbsp;&nbsp; AI Target FPS: {config.AI_TARGET_FPS}")
    st.write("---")
    
    # --- KPIs ---
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    students_detected = kpi_col1.empty()
    active_alerts = kpi_col2.empty()
    high_risk = kpi_col3.empty()
    med_risk = kpi_col4.empty()
    low_risk = kpi_col5.empty()
    st.write("---")
    
    # --- Main Layout ---
    main_col, alert_col = st.columns([3, 1])
    
    with main_col:
        st.subheader("MONITORING FEED")
        if st.session_state['input_mode'] == "UPLOADED_VIDEO":
            vid_col1, vid_col2 = st.columns(2)
            with vid_col1:
                st.markdown("**SOURCE / RAW FRAME**")
                raw_placeholder = st.empty()
            with vid_col2:
                st.markdown("**AI MONITORING FRAME**")
                ai_placeholder = st.empty()
            st.markdown("---")
            progress_placeholder = st.empty()
            info_placeholder = st.empty()
        else:
            ai_placeholder = st.empty()
            raw_placeholder = None
            progress_placeholder = None
            info_placeholder = None
            
        st.write("---")
        st.subheader("EVIDENCE")
        evidence_filters_container = st.container()
        evidence_grid_placeholder = st.empty()
        
    with alert_col:
        st.subheader("LIVE ALERTS")
        alerts_placeholder = st.empty()
        st.write("---")
        if st.button("Stop Monitoring", key="stop_monitoring_btn"):
            cleanup_session()
            st.rerun()
            
    # --- Evidence Rendering Function ---
    with evidence_filters_container:
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            sev_filter = st.selectbox("Severity", ["All", "HIGH", "MEDIUM", "LOW"], key="evidence_severity_filter")
        with f_col2:
            status_filter = st.selectbox("Status", ["All", "NEW", "REVIEWED", "DISMISSED"], key="evidence_status_filter")
            
    # ROOT CAUSE FIX: render_evidence_grid() can legitimately be called
    # more than once within a SINGLE Streamlit script execution (once
    # before the monitoring while-loop starts, and again inside the loop
    # whenever new evidence is captured). Streamlit's widget-key registry
    # is scoped to one script execution, not to one loop iteration or one
    # placeholder update, so re-rendering a button for the same event_id
    # (e.g. key=f"ev_rev_{event_id}") a second time in the same run raises
    # StreamlitDuplicateElementKey — even though the placeholder visually
    # replaces the old content. We fix this at the root by tagging every
    # widget key with a counter that increments once per render call
    # within this run, guaranteeing uniqueness no matter how many times
    # render_evidence_grid() executes in a single script pass. Because any
    # click immediately triggers st.rerun() (see below), a fresh script
    # execution starts right away and the counter naturally resets to 1 -
    # so button behaviour/state is completely unaffected.
    evidence_render_seq = {"n": 0}

    def render_evidence_grid():
        evidence_render_seq["n"] += 1
        render_seq = evidence_render_seq["n"]
        events = database.get_events(session_id=st.session_state.get('session_id'))
        with evidence_grid_placeholder.container():
            filtered_events = [e for e in events if (sev_filter == "All" or e['severity'] == sev_filter) and (status_filter == "All" or e['status'] == status_filter)]
            
            # Count summary
            h_c = sum(1 for e in filtered_events if e['severity'] == 'HIGH')
            m_c = sum(1 for e in filtered_events if e['severity'] == 'MEDIUM')
            l_c = sum(1 for e in filtered_events if e['severity'] == 'LOW')
            st.markdown(f"**Total Evidence: {len(filtered_events)}** | 🔴 HIGH: {h_c} | 🟠 MEDIUM: {m_c} | 🟢 LOW: {l_c}")
            
            # Grid
            if not filtered_events:
                st.write("No evidence matches filters.")
            else:
                cols = st.columns(3)
                for idx, e in enumerate(filtered_events):
                    col = cols[idx % 3]
                    with col:
                        ts_str = datetime.datetime.fromtimestamp(e['timestamp']).strftime("%H:%M:%S")
                        color = "🔴" if e['severity'] == "HIGH" else "🟠" if e['severity'] == "MEDIUM" else "🟢"
                        st.markdown(f"**{color} {e['severity']}**")
                        st.markdown(f"{e['event_type'].replace('_', ' ').title()}")
                        st.markdown(f"{ts_str} | Conf: {e['confidence']:.2f}")
                        if e['snapshot_path'] and os.path.exists(e['snapshot_path']):
                            st.image(e['snapshot_path'], width='stretch')
                        with st.expander("Details", expanded=False):
                            if e['status'] == 'NEW':
                                if st.button("Mark Reviewed", key=f"ev_rev_{e['event_id']}_{render_seq}"):
                                    database.update_event_status(e['event_id'], "REVIEWED")
                                    st.rerun()
                                if st.button("Dismiss", key=f"ev_dis_{e['event_id']}_{render_seq}"):
                                    database.update_event_status(e['event_id'], "DISMISSED")
                                    st.rerun()
                            else:
                                st.write(f"Status: {e['status']}")
            
    # Initial render of evidence so it's there before loop
    render_evidence_grid()
    
    # --- Monitoring Loop ---
    cap = None
    try:
        if st.session_state['input_mode'] == "LIVE_CAMERA":
            # DirectShow allows controlling BUFFERSIZE on Windows to prevent massive latency
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            st.session_state['camera_open'] = True
            st.session_state['video_open'] = False
        else:
            cap = cv2.VideoCapture(st.session_state['uploaded_video_path'])
            st.session_state['camera_open'] = False
            st.session_state['video_open'] = True
            if st.session_state['frame_idx'] > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state['frame_idx'])
                
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if st.session_state['input_mode'] == "UPLOADED_VIDEO" else 0
        
        # Initialize playback sync for video files to maintain 1x wall-clock speed
        if st.session_state['input_mode'] == "UPLOADED_VIDEO":
            st.session_state['playback_start_time'] = time.perf_counter()
            st.session_state['playback_start_frame'] = st.session_state['frame_idx']
        
        while not st.session_state['stop_monitoring']:
            
            if st.session_state['input_mode'] == "UPLOADED_VIDEO":
                elapsed_wall = time.perf_counter() - st.session_state['playback_start_time']
                expected_frame = st.session_state['playback_start_frame'] + int(elapsed_wall * fps)
                
                if expected_frame <= st.session_state['frame_idx']:
                    # Video is running faster than 1x speed, yield CPU slightly
                    time.sleep(0.01)
                else:
                    # AI is slower than video, fast-forward by grabbing frames without decoding
                    while st.session_state['frame_idx'] < expected_frame:
                        ret_skip = cap.grab()
                        if not ret_skip: break
                        st.session_state['frame_idx'] += 1

            ret, frame = cap.read()
            if not ret:
                if st.session_state['input_mode'] != "LIVE_CAMERA":
                    st.info("Video playback finished.")
                else:
                    st.error("Camera disconnected.")
                break
                
            if st.session_state['input_mode'] == "LIVE_CAMERA":
                frame = cv2.flip(frame, 1)
                
            st.session_state['frame_idx'] += 1
                    
            t0 = time.time()
            t_cap = time.perf_counter()
            
            # Analyze
            t_ai_start = time.perf_counter()
            res = st.session_state['vision'].process_frame(frame)
            t_ai_end = time.perf_counter()
            
            # Behaviors
            t_beh_start = time.perf_counter()
            confirmed_anomalies, active_highlights = st.session_state['behaviour_analyzer'].analyze_frame_data(
                res['persons'], res['phones'], res['poses'], frame.shape
            )
            t_beh_end = time.perf_counter()
            
            # Save anomalies
            evidence_added = False
            for anomaly in confirmed_anomalies:
                if config.DEBUG_MODE:
                    print(f"[BEHAVIOUR] {anomaly['event_type']} detected")
                    print(f"[CONFIRMED] {anomaly['event_type']} | Persistence satisfied")
                    print(f"[SEVERITY] {anomaly.get('severity', 'LOW')}")
                    print(f"[EVIDENCE] saving snapshot...")
                
                try:
                    snapshot_path = evidence.save_evidence(frame, anomaly, st.session_state['session_id'])
                    anomaly['snapshot_path'] = snapshot_path
                    if config.DEBUG_MODE:
                        print(f"[EVIDENCE] saved: {snapshot_path}")
                        
                    database.insert_event(anomaly, st.session_state['session_id'])
                    if config.DEBUG_MODE:
                        print(f"[DATABASE] event inserted")
                        
                    evidence_added = True
                except Exception as e:
                    print(f"[ERROR] Failed to save evidence for {anomaly['event_type']}: {e}")
                    # Do not discard the event, just insert without a path
                    anomaly['snapshot_path'] = ""
                    database.insert_event(anomaly, st.session_state['session_id'])
                
            # Draw Live View
            viz_frame = frame.copy()
            
            # 1. CONTINUOUS: Default boxes for ALL persons
            for person in res['persons']:
                x1, y1, x2, y2 = map(int, person['bbox'])
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (255, 0, 0), 1) 
                
            # 2. CONTINUOUS: Default skeletons for ALL poses
            skeleton_pairs = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
                (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
                (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
                (3, 5), (4, 6)
            ]
            from utils import get_keypoint
            for pose in res['poses']:
                kpts = pose['keypoints']
                # Draw lines
                for (i, j) in skeleton_pairs:
                    if i < len(kpts) and j < len(kpts):
                        x1, y1, c1 = get_keypoint(kpts[i])
                        x2, y2, c2 = get_keypoint(kpts[j])
                        if (c1 is None or c1 > 0.5) and (c2 is None or c2 > 0.5):
                            cv2.line(viz_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Draw points
                for kpt in kpts:
                    x, y, conf = get_keypoint(kpt)
                    if conf is None or conf > 0.5:
                        cv2.circle(viz_frame, (int(x), int(y)), 4, (0, 255, 255), -1) 
                        
            # 3. CONTINUOUS: Default boxes for ALL phones
            for phone in res['phones']:
                x1, y1, x2, y2 = map(int, phone['bbox'])
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (255, 255, 0), 1) 
                
            # 4. EVENT-DRIVEN: Active Highlights
            for highlight in active_highlights:
                bbox = highlight['bbox']
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    color = get_color(highlight['severity'])
                    cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 3) 
                    
                    label = f"{highlight['severity']} - {highlight['label']}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(viz_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                    cv2.putText(viz_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            viz_frame = cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB)
            
            if st.session_state['input_mode'] == "UPLOADED_VIDEO":
                raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_placeholder.image(raw_rgb, channels="RGB", width='stretch')
                ai_placeholder.image(viz_frame, channels="RGB", width='stretch')
                
                # Progress
                if total_frames > 0:
                    prog = st.session_state['frame_idx'] / total_frames
                    progress_placeholder.progress(min(prog, 1.0))
                    elapsed = st.session_state['frame_idx'] / fps
                    duration = total_frames / fps
                    info_placeholder.markdown(f"**Elapsed:** {format_time(elapsed)} / **Duration:** {format_time(duration)} | **Percentage:** {int(prog*100)}%")
            else:
                ai_placeholder.image(viz_frame, channels="RGB", width='stretch')
            
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
                    
            if evidence_added:
                render_evidence_grid()
                
            t_render_end = time.perf_counter()
            
            # Profiling print
            if getattr(config, "DEBUG_MODE", True):
                print(f"[PROFILE] Cap+Pre: {(t_ai_start - t_cap)*1000:.1f}ms | Vision: {(t_ai_end - t_ai_start)*1000:.1f}ms (Obj: {res.get('t_obj', 0)*1000:.1f}ms, Pose: {res.get('t_pose', 0)*1000:.1f}ms) | Behav: {(t_beh_end - t_beh_start)*1000:.1f}ms | Render: {(t_render_end - t_beh_end)*1000:.1f}ms | Total: {(t_render_end - t_cap)*1000:.1f}ms | FPS: {1.0/max(0.001, t_render_end - t_cap):.1f}")
                
            # Simple FPS target
            elapsed_time = time.time() - t0
            sleep_time = max(0, (1.0 / config.AI_TARGET_FPS) - elapsed_time)
            time.sleep(sleep_time)

    finally:
        if cap is not None:
            cap.release()
        st.session_state['camera_open'] = False
        st.session_state['video_open'] = False

def event_history_page():
    st.title("Event History")
    
    # Add Event History Filters
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        hist_sev_filter = st.selectbox("Severity", ["All", "HIGH", "MEDIUM", "LOW"], key="history_severity_filter")
    with f_col2:
        hist_status_filter = st.selectbox("Status", ["All", "NEW", "REVIEWED", "DISMISSED"], key="history_status_filter")
        
    events = database.get_events()
    filtered_events = [e for e in events if (hist_sev_filter == "All" or e['severity'] == hist_sev_filter) and (hist_status_filter == "All" or e['status'] == hist_status_filter)]
    
    if not filtered_events:
        st.write("No events recorded matching filters.")
        return
        
    for e in filtered_events:
        ts_str = datetime.datetime.fromtimestamp(e['timestamp']).strftime("%H:%M:%S")
        with st.expander(f"{ts_str} | {e['severity']} | {e['event_type']} | {e['status']}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**Severity:** {e['severity']}")
                st.write(f"**Confidence:** {e['confidence']:.2f}")
                
                if e['status'] == 'NEW':
                    if st.button("Mark Reviewed", key=f"hist_rev_{e['event_id']}"):
                        database.update_event_status(e['event_id'], "REVIEWED")
                        st.rerun()
                    if st.button("Dismiss", key=f"hist_dis_{e['event_id']}"):
                        database.update_event_status(e['event_id'], "DISMISSED")
                        st.rerun()
            with col2:
                if e['snapshot_path'] and os.path.exists(e['snapshot_path']):
                    st.image(e['snapshot_path'], width='stretch')

def main():
    st.sidebar.title("Navigation")
    
    if not st.session_state['setup_done']:
        setup_page()
    else:
        page = st.sidebar.radio("Go to", ["Live Monitoring", "Event History"])
        
        st.sidebar.markdown("---")
        if st.sidebar.button("Stop Monitoring", type="primary", width='stretch'):
            cleanup_session()
            st.rerun()
            
        if page == "Live Monitoring":
            monitoring_page()
        else:
            event_history_page()
            
    with st.sidebar.expander("Advanced / Developer Settings"):
        st.write(f"Input Source: {st.session_state.get('input_mode', 'NONE')}")
        st.write(f"Camera Resource: {'OPEN' if st.session_state.get('camera_open') else 'CLOSED'}")
        st.write(f"Video Resource: {'OPEN' if st.session_state.get('video_open') else 'CLOSED'}")
        st.write("---")
        st.write(f"Object Confidence: {config.PERSON_CONFIDENCE}")
        st.write(f"Phone Confidence: {config.PHONE_CONFIDENCE}")
        st.write(f"Analysis FPS Target: {config.AI_TARGET_FPS}")
        st.write(f"Cooldown (s): {config.EVENT_COOLDOWN_SECONDS}")

if __name__ == "__main__":
    main()
