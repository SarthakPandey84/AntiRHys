"""
Unit tests for SessionLogger.
"""

import csv
import os
from pathlib import Path
import tempfile
import threading
from session_logger import SessionLogger, CSV_HEADER


def test_session_logger_creates_file_and_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test_session_log.csv"
        logger = SessionLogger(str(log_file))

        assert log_file.exists()
        with open(log_file, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) >= 1
            assert reader[0] == CSV_HEADER


def test_session_logger_writes_event_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test_session_log.csv"
        logger = SessionLogger(str(log_file))

        success = logger.log_event(
            session_id="sess_123",
            event_type="DROWSINESS_ALERT",
            detection_state="POSSIBLE_DROWSINESS",
            eye_state="CLOSED",
            tracking_state="ACTIVE",
            ear=0.18,
            blink_count=5,
            closure_duration=1.75,
            notes="Driver closure threshold exceeded"
        )
        assert success is True

        with open(log_file, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            assert len(rows) == 2
            data_row = rows[1]
            assert data_row[1] == "sess_123"
            assert data_row[2] == "DROWSINESS_ALERT"
            assert data_row[3] == "POSSIBLE_DROWSINESS"
            assert data_row[4] == "CLOSED"
            assert data_row[6] == "0.180"
            assert data_row[7] == "5"
            assert data_row[8] == "1.75"


def test_session_logger_thread_safety():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "concurrent_log.csv"
        logger = SessionLogger(str(log_file))

        def worker(idx):
            for i in range(10):
                logger.log_event(
                    session_id=f"thread_{idx}",
                    event_type="STATE_CHANGE",
                    ear=0.25,
                    notes=f"Iteration {i}"
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(log_file, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            # Header + 50 entries
            assert len(rows) == 51
