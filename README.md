# DRISHTI MVP

AI-assisted exam monitoring system for invigilators, focusing on an event-first anomaly detection pipeline.

## Features
- **Event-First Architecture**: No permanent identity tracking. Captures specific behaviours (phone, glance, body rotation).
- **Temporal Persistence**: Anomalies must persist to trigger alerts (configurable thresholds).
- **Seat Calibration**: Maps detections to physical seat coordinates via simple click interface.
- **Evidence Snapshots**: Only ONE image is saved per confirmed event, keeping storage clean.
- **Dashboard**: Simple, non-technical interface built entirely in Streamlit.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Configure environment variables in `.env`.
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Workflow
1. Provide Exam Name & Room on the setup screen.
2. Select "Live Camera".
3. Click "Start Seat Calibration" and click the physical location of each seat (e.g. S1, S2, S3).
4. Click "Confirm Calibration".
5. In "Live Monitoring", students and objects will be detected. Confirm anomalies will generate "Live Alerts".
6. Navigate to "Event History" to review evidence images, and Mark Reviewed / Dismiss.
