import sqlite3
import config

def init_db():
    conn = sqlite3.connect(config.DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            event_type TEXT,
            severity TEXT,
            confidence REAL,
            snapshot_path TEXT,
            status TEXT,
            cooldown_group TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_event(event):
    conn = sqlite3.connect(config.DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO events (timestamp, event_type, severity, confidence, snapshot_path, status, cooldown_group)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        event['timestamp'],
        event['event_type'],
        event.get('severity', 'LOW'),
        event.get('confidence', 0.0),
        event.get('snapshot_path', ''),
        'NEW',
        event['event_type']
    ))
    conn.commit()
    event_id = c.lastrowid
    conn.close()
    return event_id

def get_events(status=None, severity=None):
    conn = sqlite3.connect(config.DB_PATH)
    c = conn.cursor()
    
    query = "SELECT * FROM events WHERE 1=1"
    params = []
    
    if status and status != "All":
        query += " AND status = ?"
        params.append(status)
    if severity and severity != "All":
        query += " AND severity = ?"
        params.append(severity)
        
    query += " ORDER BY timestamp DESC"
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    events = []
    for r in rows:
        events.append({
            "event_id": r[0],
            "timestamp": r[1],
            "event_type": r[2],
            "severity": r[3],
            "confidence": r[4],
            "snapshot_path": r[5],
            "status": r[6],
            "cooldown_group": r[7]
        })
    return events

def update_event_status(event_id, new_status):
    conn = sqlite3.connect(config.DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE events SET status = ? WHERE event_id = ?", (new_status, event_id))
    conn.commit()
    conn.close()
