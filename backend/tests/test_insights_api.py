"""
Tests for AntiRHys Phase 14 Insights, Session Logs & Adaptive Personalization APIs
===================================================================================
"""

import json
import os
import tempfile
import time
import pytest
from fastapi.testclient import TestClient

import main
from main import app, session_logger, profile_manager
from baseline_calibrator import DriverBaseline, CalibrationFailureReason
from session_logger import SessionLogger
from profile_manager import ProfileManager
from training_dataset_schema import validate_telemetry_dict


@pytest.fixture
def isolated_env(tmp_path):
    """Set up temporary log and profile files for clean, isolated testing."""
    test_log_file = str(tmp_path / "test_session_log.csv")
    test_profile_file = str(tmp_path / "test_profile.json")

    original_logger = main.session_logger
    original_profile_mgr = main.profile_manager
    original_auth = main.AUTH_TOKEN

    # Re-bind globals
    isolated_logger = SessionLogger(test_log_file)
    isolated_profile_mgr = ProfileManager(test_profile_file)

    main.session_logger = isolated_logger
    main.profile_manager = isolated_profile_mgr

    yield {
        "logger": isolated_logger,
        "profile_mgr": isolated_profile_mgr,
        "log_path": test_log_file,
        "profile_path": test_profile_file,
    }

    # Restore globals
    main.session_logger = original_logger
    main.profile_manager = original_profile_mgr
    main.AUTH_TOKEN = original_auth


@pytest.fixture
def client():
    return TestClient(app)


def test_api_sessions_empty_and_populated(client, isolated_env):
    """Verify GET /api/sessions behavior with empty and populated logs."""
    # 1. Empty state
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_sessions"] == 0
    assert data["sessions"] == []

    # 2. Populate sample session events
    logger = isolated_env["logger"]
    logger.log_event(
        session_id="test_session_101",
        event_type="SESSION_START",
        ear=0.32,
        notes="Session 101 started"
    )
    logger.log_event(
        session_id="test_session_101",
        event_type="DROWSINESS_ALERT",
        detection_state="POSSIBLE_DROWSINESS",
        ear=0.15,
        closure_duration=1.8,
        notes="Drowsiness detected"
    )
    logger.log_event(
        session_id="test_session_101",
        event_type="SESSION_END",
        notes="Session 101 ended"
    )

    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_sessions"] == 1
    session = data["sessions"][0]
    assert session["session_id"] == "test_session_101"
    assert session["total_events"] == 3
    assert session["alert_count"] == 1
    assert session["quality_status"] in ("GOOD", "FAIR")


def test_api_session_detail_and_404(client, isolated_env):
    """Verify GET /api/sessions/{session_id} returns timeline or 404."""
    # 404 on non-existent session
    resp = client.get("/api/sessions/non_existent_session")
    assert resp.status_code == 404

    # Populate and fetch valid session
    logger = isolated_env["logger"]
    logger.log_event(
        session_id="session_detail_test",
        event_type="SESSION_START",
        ear=0.30
    )
    logger.log_event(
        session_id="session_detail_test",
        event_type="STATE_CHANGE",
        detection_state="NORMAL",
        ear=0.31
    )

    resp = client.get("/api/sessions/session_detail_test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "session_detail_test"
    assert len(data["events"]) == 2
    assert "summary" in data
    assert data["summary"]["total_events"] == 2


def test_api_session_export_csv_and_json(client, isolated_env):
    """Verify GET /api/sessions/{session_id}/export for CSV and JSON formats."""
    logger = isolated_env["logger"]
    logger.log_event(
        session_id="session_export_test",
        event_type="SAMPLE",
        ear=0.28,
        perclos=0.08
    )

    # CSV Export
    resp_csv = client.get("/api/sessions/session_export_test/export?format=csv")
    assert resp_csv.status_code == 200
    assert "text/csv" in resp_csv.headers["content-type"]
    assert 'filename="antirhys_session_export_test.csv"' in resp_csv.headers.get("content-disposition", "")
    assert "session_export_test" in resp_csv.text
    assert "event_type" in resp_csv.text

    # JSON Export
    resp_json = client.get("/api/sessions/session_export_test/export?format=json")
    assert resp_json.status_code == 200
    assert "application/json" in resp_json.headers["content-type"]
    assert 'filename="antirhys_session_export_test.json"' in resp_json.headers.get("content-disposition", "")
    records = resp_json.json()
    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["session_id"] == "session_export_test"


def test_api_export_all(client, isolated_env):
    """Verify GET /api/export/all exports all sessions in CSV and JSON."""
    logger = isolated_env["logger"]
    logger.log_event(session_id="session_all_1", event_type="SESSION_START")
    logger.log_event(session_id="session_all_2", event_type="SESSION_START")

    # CSV
    resp_csv = client.get("/api/export/all?format=csv")
    assert resp_csv.status_code == 200
    assert "session_all_1" in resp_csv.text
    assert "session_all_2" in resp_csv.text

    # JSON
    resp_json = client.get("/api/export/all?format=json")
    assert resp_json.status_code == 200
    data = resp_json.json()
    assert len(data) == 2


def test_api_delete_session_and_clear_all(client, isolated_env):
    """Verify DELETE /api/sessions/{session_id} and DELETE /api/sessions."""
    logger = isolated_env["logger"]
    logger.log_event(session_id="session_to_del", event_type="SESSION_START")
    logger.log_event(session_id="session_to_keep", event_type="SESSION_START")

    # Delete single session
    resp = client.delete("/api/sessions/session_to_del")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # Confirm session_to_del is gone
    resp_list = client.get("/api/sessions")
    ids = [s["session_id"] for s in resp_list.json()["sessions"]]
    assert "session_to_del" not in ids
    assert "session_to_keep" in ids

    # 404 deleting non-existent session
    resp_del_again = client.delete("/api/sessions/session_to_del")
    assert resp_del_again.status_code == 404

    # Clear all sessions
    resp_clear = client.delete("/api/sessions")
    assert resp_clear.status_code == 200
    resp_list_after = client.get("/api/sessions")
    assert resp_list_after.json()["total_sessions"] == 0


def test_api_live_events_buffer(client, isolated_env):
    """Verify GET /api/events/live returns recent in-memory ring buffer events."""
    logger = isolated_env["logger"]
    for i in range(10):
        logger.log_event(
            session_id=f"live_sess_{i}",
            event_type="SAMPLE",
            notes=f"Event {i}"
        )

    resp = client.get("/api/events/live?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 5
    # Ring buffer returns newest events first
    assert "Event 9" in data["events"][0]["notes"]


def test_api_profile_and_reset(client, isolated_env):
    """Verify GET /api/profile and POST /api/profile/reset endpoints."""
    # Default uncalibrated profile
    resp = client.get("/api/profile")
    assert resp.status_code == 200
    p = resp.json()
    assert p["is_active"] is False
    assert p["personalized_threshold"] == 0.250
    assert p["min_threshold_bound"] == 0.180
    assert p["max_threshold_bound"] == 0.320

    # Simulate valid calibration update
    profile_mgr = isolated_env["profile_mgr"]
    valid_baseline = DriverBaseline(
        driver_id="driver_vip",
        calibrated_at=time.time(),
        is_valid=True,
        failure_reason=CalibrationFailureReason.NONE,
        sample_count=120,
        duration_sec=10.0,
        tracking_ratio=0.98,
        mean_open_ear=0.34,
        std_open_ear=0.02,
        median_open_ear=0.34,
        p10_open_ear=0.32,
        p90_open_ear=0.36,
        personalized_closure_threshold=0.272,
        personalized_perclos_threshold=0.272
    )
    ok, msg = profile_mgr.update_from_baseline(valid_baseline)
    assert ok is True

    # Check updated profile
    resp_updated = client.get("/api/profile")
    assert resp_updated.status_code == 200
    p_up = resp_updated.json()
    assert p_up["is_active"] is True
    assert p_up["personalized_threshold"] == 0.272
    assert p_up["driver_id"] == "driver_vip"

    # Reset profile via POST /api/profile/reset
    resp_reset = client.post("/api/profile/reset")
    assert resp_reset.status_code == 200
    r_data = resp_reset.json()
    assert r_data["status"] == "success"
    assert r_data["profile"]["is_active"] is False
    assert r_data["profile"]["personalized_threshold"] == 0.250


def test_api_auth_token_enforcement(client, isolated_env):
    """Verify token authorization enforcement on /api/* endpoints."""
    main.AUTH_TOKEN = "secret_anti_rhys_token"

    # 1. Without token -> 401 Unauthorized
    resp_no_auth = client.get("/api/sessions")
    assert resp_no_auth.status_code == 401

    # 2. With wrong token -> 401 Unauthorized
    resp_wrong_auth = client.get("/api/sessions?token=wrong_token")
    assert resp_wrong_auth.status_code == 401

    # 3. With query parameter token -> 200 OK
    resp_query_auth = client.get("/api/sessions?token=secret_anti_rhys_token")
    assert resp_query_auth.status_code == 200

    # 4. With Bearer token header -> 200 OK
    resp_header_auth = client.get(
        "/api/sessions",
        headers={"Authorization": "Bearer secret_anti_rhys_token"}
    )
    assert resp_header_auth.status_code == 200

    main.AUTH_TOKEN = ""


def test_api_origin_validation(client, isolated_env):
    """Verify CORS / CSWSH origin validation on /api/* endpoints."""
    # Allowed origin
    resp_allowed = client.get(
        "/api/sessions",
        headers={"Origin": "http://localhost:5173"}
    )
    assert resp_allowed.status_code == 200

    # Untrusted / spoofed origin -> 403 Forbidden
    resp_forbidden = client.get(
        "/api/sessions",
        headers={"Origin": "https://malicious-site.example.com"}
    )
    assert resp_forbidden.status_code == 403


def test_telemetry_timestamp_standardization(client, isolated_env):
    """
    Regression Test (Fix 1): Verify newly logged events expose canonical 'timestamp',
    the frontend-facing endpoints expose 'timestamp', and exported records pass the
    telemetry schema validator.
    """
    logger = isolated_env["logger"]
    logger.log_event(
        session_id="ts_standard_sess",
        event_type="DROWSINESS_ALERT",
        severity="CRITICAL",
        detection_state="POSSIBLE_DROWSINESS",
        eye_state="CLOSED",
        tracking_state="ACTIVE",
        ear=0.145,
        blink_count=8,
        closure_duration=1.75,
        perclos=0.18,
        perclos_status="VALID",
        ear_threshold=0.250,
        notes="Timestamp regression verification"
    )

    # 1. Verify in-memory live events buffer exposes 'timestamp'
    live_resp = client.get("/api/events/live?limit=1")
    assert live_resp.status_code == 200
    live_events = live_resp.json()["events"]
    assert len(live_events) == 1
    assert "timestamp" in live_events[0]
    assert len(live_events[0]["timestamp"]) > 10  # Valid ISO string
    # Both canonical 'timestamp' and backwards compatibility 'timestamp_iso' exist
    assert live_events[0]["timestamp"] == live_events[0]["timestamp_iso"]

    # 2. Verify session timeline endpoint exposes 'timestamp'
    detail_resp = client.get("/api/sessions/ts_standard_sess")
    assert detail_resp.status_code == 200
    timeline_events = detail_resp.json()["events"]
    assert len(timeline_events) == 1
    assert "timestamp" in timeline_events[0]
    assert timeline_events[0]["timestamp"] == live_events[0]["timestamp"]

    # 3. Verify JSON export exposes 'timestamp' and passes training dataset validator
    export_resp = client.get("/api/sessions/ts_standard_sess/export?format=json")
    assert export_resp.status_code == 200
    exported_events = export_resp.json()
    assert len(exported_events) == 1
    ev = exported_events[0]
    assert "timestamp" in ev

    # Validate against training dataset schema
    is_valid, err_msg = validate_telemetry_dict(ev)
    assert is_valid is True, f"Telemetry record failed schema validation: {err_msg}"


def test_legacy_12_column_and_timestamp_iso_backward_compatibility(isolated_env):
    """
    Regression Test (Fix 1): Verify pre-Phase 14 12-column logs and legacy 'timestamp_iso'
    headers remain fully readable without crashes.
    """
    import csv
    logger = isolated_env["logger"]
    legacy_header = [
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
    legacy_row = [
        "2026-10-28T12:00:00.000000",
        "legacy_session_001",
        "SESSION_START",
        "NORMAL",
        "OPEN",
        "ACTIVE",
        "0.315",
        "0",
        "0.00",
        "Legacy session started",
        "0.000",
        "CALCULATING"
    ]

    with open(logger.log_filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(legacy_header)
        writer.writerow(legacy_row)

    # Read back through SessionLogger
    records = logger._read_all_raw_rows()
    assert len(records) == 1
    rec = records[0]

    # Verify canonical 'timestamp' is populated from legacy column 0
    assert rec["timestamp"] == "2026-10-28T12:00:00.000000"
    assert rec["timestamp_iso"] == "2026-10-28T12:00:00.000000"
    assert rec["session_id"] == "legacy_session_001"
    # Default values for Phase 14 columns
    assert rec["severity"] == "INFO"
    assert rec["ear_threshold"] == 0.250
    assert rec["calibration_state"] == "IDLE"

    # Verify get_sessions aggregates legacy records correctly
    sessions = logger.get_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "legacy_session_001"
    assert sessions[0]["start_time"] == "2026-10-28T12:00:00.000000"


def test_delete_session_missing_with_trailing_blank_and_malformed_rows(client, isolated_env):
    """
    Regression Test (Fix 2): Verify SessionLogger.delete_session() and DELETE endpoint
    correctly return 404 when session does not exist, even if CSV has trailing newlines
    or malformed lines.
    """
    logger = isolated_env["logger"]
    # Write CSV with valid row, empty line, malformed line, second valid row, and trailing newline
    with open(logger.log_filepath, "w", newline="", encoding="utf-8") as f:
        f.write("timestamp,session_id,event_type,detection_state,eye_state,tracking_state,ear,blink_count,closure_duration,notes,perclos,perclos_status,severity,ear_threshold,calibration_state,detector_version\n")
        f.write("2026-10-29T08:00:00,sess_keep_1,SESSION_START,NORMAL,OPEN,ACTIVE,0.30,0,0.0,,0.0,CALCULATING,INFO,0.25,IDLE,2.1.0\n")
        f.write("\n")  # Blank row
        f.write("malformed_single_column_row\n")  # Malformed row
        f.write("2026-10-29T08:01:00,sess_keep_2,SESSION_START,NORMAL,OPEN,ACTIVE,0.30,0,0.0,,0.0,CALCULATING,INFO,0.25,IDLE,2.1.0\n")
        f.write("\n")  # Trailing blank row

    # Attempt to delete non-existent session ID
    resp = client.delete("/api/sessions/non_existent_session_id")
    assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"

    # Verify unrelated sessions and records are completely preserved
    resp_list = client.get("/api/sessions")
    assert resp_list.status_code == 200
    session_ids = [s["session_id"] for s in resp_list.json()["sessions"]]
    assert "sess_keep_1" in session_ids
    assert "sess_keep_2" in session_ids


def test_delete_session_existing_and_preservation(client, isolated_env):
    """
    Regression Test (Fix 2): Verify successful deletion of existing session preserves
    unrelated sessions.
    """
    logger = isolated_env["logger"]
    logger.log_event(session_id="target_to_remove", event_type="SESSION_START")
    logger.log_event(session_id="target_to_remove", event_type="SESSION_END")
    logger.log_event(session_id="unrelated_to_keep", event_type="SESSION_START")

    resp_del = client.delete("/api/sessions/target_to_remove")
    assert resp_del.status_code == 200
    assert resp_del.json()["status"] == "success"

    # Target is gone
    resp_detail = client.get("/api/sessions/target_to_remove")
    assert resp_detail.status_code == 404

    # Unrelated session remains intact
    resp_keep = client.get("/api/sessions/unrelated_to_keep")
    assert resp_keep.status_code == 200
    assert resp_keep.json()["session_id"] == "unrelated_to_keep"


def test_export_filename_safety_sanitization(client, isolated_env):
    """
    Regression Test (Filename safety): Verify Content-Disposition header filename is
    sanitized to prevent header injection or broken filenames from untrusted session IDs.
    """
    import urllib.parse
    logger = isolated_env["logger"]
    funky_id = 'sess_injection"quote;and spaces'
    logger.log_event(session_id=funky_id, event_type="SESSION_START")

    quoted_id = urllib.parse.quote(funky_id, safe='')
    resp = client.get(f"/api/sessions/{quoted_id}/export?format=csv")
    assert resp.status_code == 200
    cd = resp.headers.get("content-disposition", "")
    assert "attachment;" in cd
    # Verify quotes, semicolons, and spaces are sanitized in filename value
    assert 'filename="antirhys_sess_injection_quote_and_spaces.csv"' in cd


