"""
AntiRHys — Calibration Lifecycle & Bug Regression Tests
======================================================
Tests the end-to-end WebSocket calibration lifecycle, error paths,
tracking dropout rejection, sample deficiency termination, and retry semantics.

Note: Synthetic frames and landmark inputs are mocked and used strictly
for deterministic functional verification. They do not constitute real-world
driver validation.
"""

import pytest
import time
import base64
import cv2
import numpy as np
from fastapi.testclient import TestClient

from main import app
from detector import DrowsinessDetector, DetectionState, TrackingState
from baseline_calibrator import CalibrationState, CalibrationFailureReason, CalibrationConfig


def _create_mock_jpeg_b64(width: int = 64, height: int = 64) -> str:
    """Create a minimal valid base64-encoded JPEG image frame for WebSocket testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def test_calibration_start_request_and_response_format():
    """Verify that a frame-free start_calibration command receives an immediate structured response."""
    client = TestClient(app)
    with client.websocket_connect("/ws/detect") as ws:
        # Send start_calibration command
        req = {
            "command": "start_calibration",
            "driver_id": "test_driver_01",
            "session_id": "client_session_abc",
            "client_timestamp": 12345678.0
        }
        ws.send_json(req)
        resp = ws.receive_json()

        assert resp.get("status") == "success"
        assert resp.get("command") == "start_calibration"
        assert resp.get("calibration_state") == "COLLECTING"
        assert resp.get("calibration_progress") == 0.0
        assert resp.get("calibration_failure_reason") == "NONE"
        assert resp.get("client_session_id") == "client_session_abc"
        assert resp.get("client_timestamp") == 12345678.0
        assert "error" not in resp


def test_calibration_with_valid_tracking_samples_completes():
    """
    Verify that an attentive driver with valid eye landmarks successfully completes
    calibration after target_duration_sec, establishes a personalized threshold,
    and transitions state to COMPLETED.
    """
    detector = DrowsinessDetector("test_calib_valid")
    detector.start_calibration(driver_id="driver_valid", start_timestamp=100.0)
    assert detector.calibrator.state == CalibrationState.COLLECTING

    # Typical attentive open eye (EAR ≈ 0.35)
    OPEN_EYE = [
        (10.0, 50.0), (16.0, 43.0), (24.0, 43.0),
        (30.0, 50.0), (24.0, 50.0), (16.0, 50.0)
    ]

    # Feed 120 samples at 10 FPS from t=100.0 to t=112.0s (12.0s elapsed)
    now = 100.0
    for i in range(121):
        now = 100.0 + (i * 0.1)
        res = detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=OPEN_EYE,
            right_eye_points=OPEN_EYE,
            now=now
        )

    # After 12.0 seconds, calibration must be finalized and COMPLETED
    assert res.calibration_state == "COMPLETED"
    assert res.calibration_progress == 1.0
    assert res.baseline_active is True
    # Default open eye 0.35 * 0.70 = 0.245
    assert res.calibrated_ear_threshold == pytest.approx(0.245, abs=0.01)
    assert detector.config.ear_threshold == pytest.approx(0.245, abs=0.01)


def test_calibration_tracking_unavailable_rejection():
    """
    Verify that if face tracking is lost during calibration (e.g. camera pointed away),
    calibration fails with LOW_TRACKING_RATIO and restores the standard threshold (0.25).
    """
    detector = DrowsinessDetector("test_calib_unavail")
    detector.start_calibration(driver_id="driver_unavail", start_timestamp=200.0)
    assert detector.calibrator.state == CalibrationState.COLLECTING

    # Feed frames with no face detected for 2.5 seconds (exceeding tracking_lost_timeout_sec)
    now = 200.0
    for i in range(25):
        now = 200.0 + (i * 0.1)
        res = detector.process_frame_landmarks(
            has_face=False,
            now=now
        )

    # Tracking timeout (2.0s) should have aborted the calibrator with LOW_TRACKING_RATIO
    assert res.calibration_state == "FAILED"
    assert res.calibration_failure_reason == "LOW_TRACKING_RATIO"
    assert res.baseline_active is False
    assert res.calibrated_ear_threshold == 0.250


def test_calibration_insufficient_samples_does_not_get_stuck():
    """
    CRITICAL REGRESSION TEST:
    Verify that if the duration reaches target_duration_sec (12.0s) but the number of
    collected samples is below min_valid_samples (50), calibration does NOT remain
    silently stuck in COLLECTING (100%), but terminates cleanly and transitions to FAILED
    with INSUFFICIENT_SAMPLES.
    """
    detector = DrowsinessDetector("test_insufficient_samples")
    detector.start_calibration(driver_id="driver_low_fps", start_timestamp=300.0)
    assert detector.calibrator.state == CalibrationState.COLLECTING

    OPEN_EYE = [
        (10.0, 50.0), (16.0, 43.0), (24.0, 43.0),
        (30.0, 50.0), (24.0, 50.0), (16.0, 50.0)
    ]

    # Feed only 25 samples over 12.0 seconds (e.g. ~2 FPS severe frame drop)
    # Target duration is 12.0s; min_valid_samples is 50.
    for i in range(25):
        now = 300.0 + (i * 0.5)  # 0.5s interval -> total 12.0s
        res = detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=OPEN_EYE,
            right_eye_points=OPEN_EYE,
            now=now
        )

    # Must finalize after 12.0s and fail with INSUFFICIENT_SAMPLES instead of remaining stuck
    assert res.calibration_state == "FAILED"
    assert res.calibration_failure_reason == "INSUFFICIENT_SAMPLES"
    assert res.baseline_active is False
    assert res.calibrated_ear_threshold == 0.250


def test_calibration_aborted_by_command():
    """Verify that abort_calibration command cancels an active calibration immediately."""
    client = TestClient(app)
    with client.websocket_connect("/ws/detect") as ws:
        # 1. Start calibration
        ws.send_json({"command": "start_calibration", "driver_id": "test_abort"})
        resp1 = ws.receive_json()
        assert resp1.get("calibration_state") == "COLLECTING"

        # 2. Abort calibration
        ws.send_json({"command": "abort_calibration"})
        resp2 = ws.receive_json()
        assert resp2.get("calibration_state") == "FAILED"
        assert resp2.get("calibration_failure_reason") == "USER_ABORTED"
        assert resp2.get("baseline_active") is False


def test_calibration_repeated_start_and_retry_after_failure():
    """
    Verify that repeated start_calibration requests cleanly restart the state machine
    and that retrying after a failed attempt succeeds without requiring a fresh session.
    """
    detector = DrowsinessDetector("test_retry")

    # Attempt 1: Fails due to lack of face tracking
    detector.start_calibration(driver_id="driver_retry", start_timestamp=400.0)
    for i in range(25):
        detector.process_frame_landmarks(has_face=False, now=400.0 + i * 0.1)
    assert detector.calibrator.state == CalibrationState.FAILED

    # Attempt 2 (Retry): Re-invoking start_calibration must cleanly reset state to COLLECTING
    detector.start_calibration(driver_id="driver_retry", start_timestamp=500.0)
    assert detector.calibrator.state == CalibrationState.COLLECTING
    assert detector.calibrator.failure_reason == CalibrationFailureReason.NONE

    OPEN_EYE = [
        (10.0, 50.0), (16.0, 43.0), (24.0, 43.0),
        (30.0, 50.0), (24.0, 50.0), (16.0, 50.0)
    ]
    # Feed valid frames
    for i in range(121):
        res = detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=OPEN_EYE,
            right_eye_points=OPEN_EYE,
            now=500.0 + i * 0.1
        )
    assert res.calibration_state == "COMPLETED"
    assert res.baseline_active is True


def test_calibration_websocket_repeated_clicks_protection():
    """Verify that multiple consecutive start_calibration commands do not throw or corrupt state."""
    client = TestClient(app)
    with client.websocket_connect("/ws/detect") as ws:
        for _ in range(5):
            ws.send_json({"command": "start_calibration", "driver_id": "rapid_clicker"})
            resp = ws.receive_json()
            assert resp.get("status") == "success"
            assert resp.get("calibration_state") == "COLLECTING"
            assert resp.get("calibration_progress") == 0.0
