# Drishti: AI Exam Monitoring System

Drishti is an AI-powered examination monitoring tool designed for invigilators. Instead of permanently tracking student identities, it uses an event-first architecture to detect specific anomalous behaviors like phone usage, irregular glancing, and excessive body rotation. 

This project was built to provide a reliable, low-overhead way to monitor exam halls without excessive storage requirements or complex setups.

## Key Features

* **Event-First Detection**: Focuses on specific actions rather than continuous tracking.
* **Temporal Persistence**: Anomalies need to persist over a configurable threshold to trigger alerts, minimizing false positives.
* **Seat Calibration**: A simple click interface to map physical seat coordinates to the camera feed.
* **Efficient Evidence Storage**: Saves only a single snapshot per confirmed event to keep disk usage low.
* **Streamlit Dashboard**: A straightforward, non-technical UI for real-time monitoring and evidence review.

## Tech Stack

* **Frontend/UI**: Streamlit
* **Computer Vision**: Ultralytics YOLO (v8), OpenCV
* **Data Management**: SQLite, Pandas

## Getting Started

### Prerequisites
Make sure you have Python 3.8 or higher installed. It's recommended to use a virtual environment.

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repo-url>
   cd DRISHTI_MVP
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up your environment variables by copying the example file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to configure your specific settings if needed.

### Running the App

Start the Streamlit server:
```bash
streamlit run app.py
```
The application will open automatically in your default web browser.

## Usage Workflow

1. **Initial Setup**: Enter the Exam Name and Room number on the startup screen.
2. **Select Source**: Choose either "Live Camera" or upload a pre-recorded video.
3. **Seat Calibration**: 
   * Click "Start Seat Calibration".
   * Click on the physical locations of the seats in the video feed (e.g., S1, S2, S3).
   * Click "Confirm Calibration" to save the mapping.
4. **Monitoring**: In the "Live Monitoring" view, the system will track students and objects. Persistent anomalies will appear under "Live Alerts".
5. **Review Evidence**: Go to the "Event History" tab to look at captured evidence snapshots. You can mark events as "Reviewed" or "Dismiss" them as necessary.

## Project Structure Overview

* `app.py`: Main Streamlit application entry point.
* `vision.py`: Core computer vision inference and object detection logic.
* `behaviour.py`: Analyzes temporal behavior and triggers anomaly events.
* `severity.py`: Defines rules for anomaly severity classification.
* `config.py` / `.env`: Configuration management.
* `database.py`: SQLite database operations for logging events.
* `evidence.py`: Manages the saving and retrieval of evidence snapshots.
