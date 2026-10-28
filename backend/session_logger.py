"""
AntiRHys Session Logger
=======================
Provides structured, thread-safe CSV logging for driver monitoring sessions.

Schema:
- timestamp_iso: ISO 8601 UTC/local timestamp
- session_id: Unique session identifier
- event_type: SESSION_START, STATE_CHANGE, TRACKING_LOST, DROWSINESS_ALERT, RECOVERY, SESSION_END, ERROR
- detection_state: NORMAL, POSSIBLE_DROWSINESS, TRACKING_LOST, UNAVAILABLE
- eye_state: OPEN, CLOSED, UNKNOWN
- tracking_state: ACTIVE, LOST, SEARCHING
- ear: Calculated average Eye Aspect Ratio (or 0.0)
- blink_count: Total cumulative blinks in session
- closure_duration: Duration eyes were closed (seconds)
- notes: Contextual diagnostic string
"""

import csv
from datetime import datetime
import math
import os
from pathlib import Path
import threading
from typing import Optional


CSV_HEADER = [
    "timestamp_iso",
    "session_id",
    "event_type",
    "detection_state",
    "eye_state",
    "tracking_state",
    "ear",
    "blink_count",
    "closure_duration",
    "notes",
    "perclos",
    "perclos_status"
]


class SessionLogger:
    """Thread-safe CSV logger for driver monitoring sessions."""

    def __init__(self, log_filepath: str = "session_log.csv"):
        self.log_filepath = Path(log_filepath)
        self._lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self) -> None:
        """Create the log file with header if it doesn't already exist."""
        with self._lock:
            try:
                file_exists = self.log_filepath.exists() and self.log_filepath.stat().st_size > 0
                if not file_exists:
                    self.log_filepath.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.log_filepath, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(CSV_HEADER)
            except Exception as e:
                print(f"[SessionLogger] Warning: failed to ensure CSV header: {e}")

    def log_event(
        self,
        session_id: str,
        event_type: str,
        detection_state: str = "NORMAL",
        eye_state: str = "UNKNOWN",
        tracking_state: str = "ACTIVE",
        ear: float = 0.0,
        blink_count: int = 0,
        closure_duration: float = 0.0,
        notes: str = "",
        perclos: float = 0.0,
        perclos_status: str = "CALCULATING"
    ) -> bool:
        """
        Record an event to the CSV log file safely without raising exceptions.

        Returns:
            bool: True if write succeeded, False otherwise.
        """
        now_iso = datetime.now().isoformat()
        clean_ear = ear if math.isfinite(ear) else 0.0
        clean_duration = closure_duration if math.isfinite(closure_duration) else 0.0
        clean_perclos = perclos if math.isfinite(perclos) else 0.0

        row = [
            now_iso,
            session_id,
            event_type,
            detection_state,
            eye_state,
            tracking_state,
            f"{clean_ear:.3f}",
            blink_count,
            f"{clean_duration:.2f}",
            notes.replace("\n", " "),  # Sanitize newlines
            f"{clean_perclos:.3f}",
            perclos_status
        ]

        try:
            with self._lock:
                with open(self.log_filepath, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            return True
        except Exception as e:
            # Must not crash detection pipeline on disk write failure
            print(f"[SessionLogger] Write error: {e}")
            return False
