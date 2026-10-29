"""
AntiRHys Session Logger & Event Persistence Engine
==================================================
Provides structured, thread-safe, bounded CSV/in-memory logging,
session history aggregation, export utilities, and live telemetry feeds.

Phase 14 Structured Event Schema (16 columns):
- timestamp_iso: ISO 8601 UTC/local timestamp
- session_id: Unique session identifier
- event_type: SESSION_START, STATE_CHANGE, TRACKING_LOST, DROWSINESS_ALERT, RECOVERY, SESSION_END, ERROR, CALIBRATION_START, CALIBRATION_SUCCESS, CALIBRATION_FAILED, BASELINE_APPLIED, BASELINE_CLEARED
- detection_state: NORMAL, POSSIBLE_DROWSINESS, TRACKING_LOST, UNAVAILABLE, INITIALIZING
- eye_state: OPEN, CLOSED, UNKNOWN
- tracking_state: ACTIVE, LOST, SEARCHING
- ear: Calculated Eye Aspect Ratio (or 0.0)
- blink_count: Total cumulative blinks in session
- closure_duration: Confirmed closed duration (seconds)
- notes: Contextual diagnostic string
- perclos: Calculated PERCLOS ratio
- perclos_status: Status string (CALCULATING, VALID, INSUFFICIENT_DATA, UNAVAILABLE)
- severity: INFO, WARNING, CRITICAL, ERROR
- ear_threshold: Active eye-closure threshold
- calibration_state: IDLE, COLLECTING, COMPLETED, FAILED
- detector_version: Version of detection software/configuration
"""

import collections
import csv
from datetime import datetime
import io
import math
import os
from pathlib import Path
import threading
from typing import List, Dict, Any, Optional

DEFAULT_DETECTOR_VERSION = "2.1.0"
MAX_LOG_FILE_BYTES = 10 * 1024 * 1024  # 10 MB bound to prevent unbounded disk growth
MAX_LIVE_BUFFER_EVENTS = 250

CSV_HEADER = [
    "timestamp",
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
    "perclos_status",
    "severity",
    "ear_threshold",
    "calibration_state",
    "detector_version",
]


def infer_event_severity(event_type: str) -> str:
    """Infer structured severity level if not explicitly provided."""
    upper = event_type.upper()
    if "DROWSINESS_ALERT" in upper or "CRITICAL" in upper:
        return "CRITICAL"
    if "ERROR" in upper:
        return "ERROR"
    if "TRACKING_LOST" in upper or "CALIBRATION_FAILED" in upper or "WARNING" in upper:
        return "WARNING"
    return "INFO"


class SessionLogger:
    """Thread-safe, bounded event logger and session history repository."""

    def __init__(self, log_filepath: str = "session_log.csv"):
        self.log_filepath = Path(log_filepath)
        self._lock = threading.Lock()
        self._live_events = collections.deque(maxlen=MAX_LIVE_BUFFER_EVENTS)
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

    def _check_rotation(self) -> None:
        """Prune oldest records if the file size exceeds MAX_LOG_FILE_BYTES."""
        try:
            if not self.log_filepath.exists():
                return
            size = self.log_filepath.stat().st_size
            if size <= MAX_LOG_FILE_BYTES:
                return

            # Keep the header and newest 60% of lines
            with open(self.log_filepath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            if len(lines) <= 100:
                return

            header = lines[0]
            cutoff = int(len(lines) * 0.4)  # discard oldest 40%
            preserved_lines = [header] + lines[cutoff:]

            with open(self.log_filepath, "w", encoding="utf-8", newline="") as f:
                f.writelines(preserved_lines)
            print(f"[SessionLogger] Rotated log file: pruned {cutoff} oldest rows.")
        except Exception as e:
            print(f"[SessionLogger] Rotation warning: {e}")

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
        perclos_status: str = "CALCULATING",
        severity: Optional[str] = None,
        ear_threshold: Optional[float] = None,
        calibration_state: Optional[str] = None,
        detector_version: str = DEFAULT_DETECTOR_VERSION,
    ) -> bool:
        """
        Record a structured event safely to disk and in-memory live buffer.
        Never throws unhandled exceptions that could interrupt detection.
        """
        now_iso = datetime.now().isoformat()
        clean_ear = ear if (isinstance(ear, (int, float)) and math.isfinite(ear)) else 0.0
        clean_duration = closure_duration if (isinstance(closure_duration, (int, float)) and math.isfinite(closure_duration)) else 0.0
        clean_perclos = perclos if (isinstance(perclos, (int, float)) and math.isfinite(perclos)) else 0.0
        clean_thresh = ear_threshold if (isinstance(ear_threshold, (int, float)) and math.isfinite(ear_threshold)) else 0.250
        event_severity = severity or infer_event_severity(event_type)
        calib_state = calibration_state or "IDLE"

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
            notes.replace("\n", " ").replace("\r", " "),  # Sanitize newlines
            f"{clean_perclos:.3f}",
            perclos_status,
            event_severity,
            f"{clean_thresh:.3f}",
            calib_state,
            detector_version,
        ]

        # In-memory dictionary representation for live API
        event_dict = {
            "timestamp": now_iso,
            "timestamp_iso": now_iso,  # Backwards compatibility alias
            "session_id": session_id,
            "event_type": event_type,
            "detection_state": detection_state,
            "eye_state": eye_state,
            "tracking_state": tracking_state,
            "ear": round(clean_ear, 3),
            "blink_count": blink_count,
            "closure_duration": round(clean_duration, 2),
            "notes": notes,
            "perclos": round(clean_perclos, 3),
            "perclos_status": perclos_status,
            "severity": event_severity,
            "ear_threshold": round(clean_thresh, 3),
            "calibration_state": calib_state,
            "detector_version": detector_version,
        }

        with self._lock:
            self._live_events.append(event_dict)
            try:
                self._check_rotation()
                with open(self.log_filepath, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                return True
            except Exception as e:
                print(f"[SessionLogger] Write error: {e}")
                return False

    def get_live_events(self, limit: int = 100, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent events from in-memory ring buffer (fast, non-blocking, newest first)."""
        with self._lock:
            events = list(self._live_events)
        if session_id:
            events = [e for e in events if e.get("session_id") == session_id]
        return list(reversed(events[-limit:]))

    def _read_all_raw_rows(self) -> List[Dict[str, Any]]:
        """Read all rows from CSV with backwards-compatible column handling."""
        if not self.log_filepath.exists():
            return []

        rows: List[Dict[str, Any]] = []
        with self._lock:
            try:
                with open(self.log_filepath, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        return []

                    for line in reader:
                        if not line or len(line) < 2:
                            continue

                        # Backwards compatibility: fill default values for columns added in Phase 14
                        t_iso = line[0] if len(line) > 0 else ""
                        sid = line[1] if len(line) > 1 else "unknown"
                        etype = line[2] if len(line) > 2 else "UNKNOWN"
                        dstate = line[3] if len(line) > 3 else "NORMAL"
                        estate = line[4] if len(line) > 4 else "UNKNOWN"
                        tstate = line[5] if len(line) > 5 else "ACTIVE"
                        try:
                            ear_val = float(line[6]) if len(line) > 6 else 0.0
                        except ValueError:
                            ear_val = 0.0
                        try:
                            blinks = int(line[7]) if len(line) > 7 else 0
                        except ValueError:
                            blinks = 0
                        try:
                            cdur = float(line[8]) if len(line) > 8 else 0.0
                        except ValueError:
                            cdur = 0.0
                        notes = line[9] if len(line) > 9 else ""
                        try:
                            perclos_val = float(line[10]) if len(line) > 10 else 0.0
                        except ValueError:
                            perclos_val = 0.0
                        pstatus = line[11] if len(line) > 11 else "CALCULATING"

                        # Extended Phase 14 columns
                        severity = line[12] if len(line) > 12 and line[12] else infer_event_severity(etype)
                        try:
                            thresh = float(line[13]) if len(line) > 13 and line[13] else 0.250
                        except ValueError:
                            thresh = 0.250
                        calib_state = line[14] if len(line) > 14 and line[14] else "IDLE"
                        det_version = line[15] if len(line) > 15 and line[15] else DEFAULT_DETECTOR_VERSION

                        rows.append({
                            "timestamp": t_iso,
                            "timestamp_iso": t_iso,  # Backwards compatibility alias
                            "session_id": sid,
                            "event_type": etype,
                            "detection_state": dstate,
                            "eye_state": estate,
                            "tracking_state": tstate,
                            "ear": round(ear_val, 3),
                            "blink_count": blinks,
                            "closure_duration": round(cdur, 2),
                            "notes": notes,
                            "perclos": round(perclos_val, 3),
                            "perclos_status": pstatus,
                            "severity": severity,
                            "ear_threshold": round(thresh, 3),
                            "calibration_state": calib_state,
                            "detector_version": det_version,
                        })
            except Exception as e:
                print(f"[SessionLogger] Error reading log file: {e}")
        return rows

    def get_sessions(self) -> List[Dict[str, Any]]:
        """
        Aggregate all logged events by session_id and calculate summary metrics,
        data quality indicators, and duration.
        """
        raw_events = self._read_all_raw_rows()
        if not raw_events:
            return []

        sessions_map: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for ev in raw_events:
            sessions_map[ev["session_id"]].append(ev)

        summary_list: List[Dict[str, Any]] = []

        for sid, evs in sessions_map.items():
            start_iso = evs[0].get("timestamp") or evs[0].get("timestamp_iso", "")
            end_iso = evs[-1].get("timestamp") or evs[-1].get("timestamp_iso", "")

            # Calculate approximate duration from timestamps
            duration_sec = 0.0
            try:
                t0 = datetime.fromisoformat(start_iso)
                t1 = datetime.fromisoformat(end_iso)
                duration_sec = max(0.0, (t1 - t0).total_seconds())
            except Exception:
                duration_sec = 0.0

            alert_count = sum(1 for e in evs if e["event_type"] == "DROWSINESS_ALERT")
            error_count = sum(1 for e in evs if e["severity"] == "ERROR" or e["event_type"] == "ERROR")
            calib_events = [e for e in evs if "CALIBRATION" in e["event_type"]]
            calibrated = any(e["event_type"] == "CALIBRATION_SUCCESS" or e["calibration_state"] == "COMPLETED" for e in evs)

            # EAR statistics over active frames
            active_ears = [e["ear"] for e in evs if e["tracking_state"] == "ACTIVE" and e["ear"] > 0.0]
            mean_ear = round(sum(active_ears) / len(active_ears), 3) if active_ears else 0.0

            # PERCLOS statistics
            valid_perclos = [e["perclos"] for e in evs if e["perclos_status"] == "VALID"]
            mean_perclos = round(sum(valid_perclos) / len(valid_perclos), 3) if valid_perclos else 0.0

            # Tracking coverage calculation
            tracked_events = sum(1 for e in evs if e["tracking_state"] == "ACTIVE")
            coverage_ratio = round(tracked_events / len(evs), 3) if evs else 0.0

            if coverage_ratio >= 0.80:
                quality_status = "GOOD"
            elif coverage_ratio >= 0.50:
                quality_status = "FAIR"
            else:
                quality_status = "INSUFFICIENT_TRACKING"

            # Last calibrated threshold seen in this session
            calibrated_threshold = None
            for e in reversed(evs):
                if e.get("ear_threshold") and e.get("ear_threshold") != 0.250:
                    calibrated_threshold = e["ear_threshold"]
                    break

            summary_list.append({
                "session_id": sid,
                "start_time": start_iso,
                "end_time": end_iso,
                "duration_sec": round(duration_sec, 1),
                "total_events": len(evs),
                "alert_count": alert_count,
                "error_count": error_count,
                "mean_ear": mean_ear,
                "mean_perclos": mean_perclos,
                "tracking_coverage_ratio": coverage_ratio,
                "quality_status": quality_status,
                "calibrated": calibrated,
                "calibrated_threshold": calibrated_threshold,
                "has_incomplete_data": quality_status == "INSUFFICIENT_TRACKING",
            })

        # Sort by start_time descending (newest sessions first)
        summary_list.sort(key=lambda s: s["start_time"], reverse=True)
        return summary_list

    def get_session_timeline(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full structured event timeline and quality summary for a specific session."""
        raw_events = self._read_all_raw_rows()
        evs = [e for e in raw_events if e["session_id"] == session_id]
        if not evs:
            return None

        sessions = self.get_sessions()
        summary = next((s for s in sessions if s["session_id"] == session_id), None)

        return {
            "session_id": session_id,
            "summary": summary,
            "events": evs,
        }

    def export_session_csv(self, session_id: str) -> str:
        """Generate a standalone CSV string for a specific session."""
        evs = [e for e in self._read_all_raw_rows() if e["session_id"] == session_id]
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(CSV_HEADER)
        for e in evs:
            writer.writerow([e.get(col, "") for col in CSV_HEADER])
        return output.getvalue()

    def export_all_csv(self) -> str:
        """Generate a complete CSV string for all recorded sessions."""
        evs = self._read_all_raw_rows()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(CSV_HEADER)
        for e in evs:
            writer.writerow([e.get(col, "") for col in CSV_HEADER])
        return output.getvalue()

    def export_session_json(self, session_id: str) -> List[Dict[str, Any]]:
        """Export session events as structured JSON list."""
        return [e for e in self._read_all_raw_rows() if e["session_id"] == session_id]

    def export_all_json(self) -> List[Dict[str, Any]]:
        """Export all events across all sessions as structured JSON list."""
        return self._read_all_raw_rows()

    def delete_session(self, session_id: str) -> bool:
        """Safely remove all records associated with session_id from disk and memory."""
        with self._lock:
            # Clear from in-memory live buffer
            pre_live_count = len(self._live_events)
            self._live_events = collections.deque(
                [e for e in self._live_events if e.get("session_id") != session_id],
                maxlen=MAX_LIVE_BUFFER_EVENTS,
            )
            deleted_from_live = (len(self._live_events) < pre_live_count)

            if not self.log_filepath.exists():
                return deleted_from_live

            try:
                with open(self.log_filepath, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        return deleted_from_live
                    all_rows = list(reader)

                # Check whether target session ID actually exists on disk
                deleted_from_disk = any(len(row) > 1 and row[1] == session_id for row in all_rows)

                if not deleted_from_disk and not deleted_from_live:
                    return False

                if deleted_from_disk:
                    preserved_rows = [row for row in all_rows if not (len(row) > 1 and row[1] == session_id)]
                    with open(self.log_filepath, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(preserved_rows)

                return True
            except Exception as e:
                print(f"[SessionLogger] Error deleting session {session_id}: {e}")
                return False

    def clear_all_logs(self) -> bool:
        """User-initiated complete purge of all session log records."""
        with self._lock:
            self._live_events.clear()
            try:
                with open(self.log_filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADER)
                return True
            except Exception as e:
                print(f"[SessionLogger] Error clearing all logs: {e}")
                return False
