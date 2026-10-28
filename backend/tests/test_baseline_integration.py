"""
AntiRHys Personalized Baseline Integration Tests
================================================
Verifies end-to-end integration between DrowsinessDetector, BaselineCalibrator,
and runtime threshold consumption:
1. Live frame EAR observations are directly ingested by the active calibrator.
2. Calibrated threshold is actively consumed by the detector and PERCLOS calculator.
3. Live state machine transitions adapt to the personalized threshold.
4. Safe fallback to default uncalibrated threshold when calibration fails or is cleared.
5. Corrupt file persistence safety.
6. WebSocket control commands (start_calibration, clear_baseline).
"""

import json
import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from detector import (
    DrowsinessDetector,
    DetectionConfig,
    DetectionState,
    AlarmState,
    EyeState,
)
from baseline_calibrator import (
    BaselineCalibrator,
    CalibrationConfig,
    CalibrationState,
    CalibrationFailureReason,
    DriverBaseline,
    save_baseline_to_file,
    load_baseline_from_file,
)
from main import app

# Standard synthetic landmarks
OPEN_EYE_WIDE = [
    (10.0, 50.0), (16.0, 42.0), (24.0, 42.0),
    (30.0, 50.0), (24.0, 52.0), (16.0, 52.0)
]  # EAR ≈ 0.50

OPEN_EYE_NORMAL = [
    (10.0, 50.0), (16.0, 43.0), (24.0, 43.0),
    (30.0, 50.0), (24.0, 50.0), (16.0, 50.0)
]  # EAR ≈ 0.35

CLOSED_EYE = [
    (10.0, 50.0), (16.0, 49.5), (24.0, 49.5),
    (30.0, 50.0), (24.0, 50.5), (16.0, 50.5)
]  # EAR ≈ 0.05


def test_baseline_actively_consumed_by_live_detector():
    """
    Verify that once calibrated, the detector's live state machine
    actually uses the personalized threshold instead of the default.
    """
    config = DetectionConfig(ear_threshold=0.25)
    detector = DrowsinessDetector("test_consume", config)

    # Before calibration: default threshold is 0.25
    assert detector.config.ear_threshold == 0.25
    assert detector.baseline is None

    # Start calibration with a 10s target
    calib_cfg = CalibrationConfig(
        min_duration_sec=8.0,
        target_duration_sec=10.0,
        min_valid_samples=50,
        closure_multiplier=0.70,
        min_closure_threshold=0.18,
        max_closure_threshold=0.32,
    )
    start_t = 1000.0
    detector.start_calibration(driver_id="driver_vip", config=calib_cfg, start_timestamp=start_t)
    assert detector.calibrator is not None
    assert detector.calibrator.state == CalibrationState.COLLECTING

    # Feed 100 frames of attentive open eyes with EAR ≈ 0.35
    for i in range(100):
        t = start_t + (i * 0.1)
        res = detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=OPEN_EYE_NORMAL,
            right_eye_points=OPEN_EYE_NORMAL,
            now=t
        )
        assert res.calibration_state in ("COLLECTING", "COMPLETED")

    # Finalize calibration
    baseline = detector.finalize_calibration(now=start_t + 10.0)
    assert baseline.is_valid is True
    assert detector.baseline is not None
    assert detector.baseline.is_valid is True

    # The calibrated threshold is ≈ 0.35 * 0.70 = 0.245
    expected_thresh = baseline.personalized_closure_threshold
    assert detector.config.ear_threshold == expected_thresh
    assert detector.perclos_calculator.closure_ear_threshold == expected_thresh

    # Test an intermediate EAR value: e.g. EAR = 0.23
    # Synthetic eye with height around 4.6 (giving EAR ≈ 0.23)
    INTERMEDIATE_EYE = [
        (10.0, 50.0), (16.0, 45.4), (24.0, 45.4),
        (30.0, 50.0), (24.0, 50.0), (16.0, 50.0)
    ]
    # At EAR ≈ 0.23:
    # Under factory default (0.25), this is CLOSED.
    # Under a personalized threshold of 0.20, it would be OPEN.
    # Let's apply a custom lower baseline to prove consumption!
    low_baseline = DriverBaseline(
        driver_id="narrow_eyes",
        calibrated_at=1000.0,
        is_valid=True,
        failure_reason=CalibrationFailureReason.NONE,
        sample_count=80,
        duration_sec=10.0,
        tracking_ratio=1.0,
        mean_open_ear=0.28,
        std_open_ear=0.02,
        median_open_ear=0.28,
        p10_open_ear=0.25,
        p90_open_ear=0.31,
        personalized_closure_threshold=0.19,  # 0.19 closure threshold!
        personalized_perclos_threshold=0.15,
    )
    detector.apply_baseline(low_baseline)
    assert detector.config.ear_threshold == 0.19

    # Now feed INTERMEDIATE_EYE (EAR ≈ 0.23)
    res = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=INTERMEDIATE_EYE,
        right_eye_points=INTERMEDIATE_EYE,
        now=2000.0
    )
    # Since 0.23 > 0.19, eyes are OPEN!
    assert res.eye_state == EyeState.OPEN
    assert res.baseline_active is True
    assert res.calibrated_ear_threshold == 0.19

    # Clear baseline and verify it reverts to default 0.25
    detector.clear_baseline()
    assert detector.config.ear_threshold == 0.25
    assert detector.baseline is None

    # Now under default 0.25, the same INTERMEDIATE_EYE (0.23) is classified as CLOSED!
    res_after = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=INTERMEDIATE_EYE,
        right_eye_points=INTERMEDIATE_EYE,
        now=2001.0
    )
    assert res_after.eye_state == EyeState.CLOSED
    assert res_after.baseline_active is False
    assert res_after.calibrated_ear_threshold == 0.25


def test_fallback_when_calibration_fails():
    """Verify that failed calibration leaves the detector in safe default fallback."""
    detector = DrowsinessDetector("test_fallback")
    detector.start_calibration()

    # Feed frames with eyes closed (drowsiness during calibration)
    start_t = 100.0
    for i in range(80):
        detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=CLOSED_EYE,
            right_eye_points=CLOSED_EYE,
            now=start_t + (i * 0.1)
        )

    baseline = detector.finalize_calibration(now=start_t + 8.0)
    assert baseline.is_valid is False
    assert detector.baseline is None
    # Still at factory default threshold
    assert detector.config.ear_threshold == 0.25


def test_file_persistence_safe_handling():
    """Test safe persistence, round-trip, and corruption resilience."""
    baseline = DriverBaseline(
        driver_id="driver_safe",
        calibrated_at=1234567.89,
        is_valid=True,
        failure_reason=CalibrationFailureReason.NONE,
        sample_count=100,
        duration_sec=12.0,
        tracking_ratio=0.98,
        mean_open_ear=0.345,
        std_open_ear=0.021,
        median_open_ear=0.342,
        p10_open_ear=0.312,
        p90_open_ear=0.378,
        personalized_closure_threshold=0.239,
        personalized_perclos_threshold=0.180,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "baseline.json")

        # 1. Successful save & load
        assert save_baseline_to_file(baseline, save_path) is True
        loaded = load_baseline_from_file(save_path)
        assert loaded is not None
        assert loaded.driver_id == "driver_safe"
        assert loaded.personalized_closure_threshold == 0.239

        # 2. Corrupt file: not JSON
        with open(save_path, "w") as f:
            f.write("THIS_IS_NOT_VALID_JSON{:::}")
        assert load_baseline_from_file(save_path) is None

        # 3. Corrupt file: missing required fields
        with open(save_path, "w") as f:
            json.dump({"driver_id": "test"}, f)
        assert load_baseline_from_file(save_path) is None

        # 4. Corrupt file: non-finite NaN/Inf
        with open(save_path, "w") as f:
            f.write('{"mean_open_ear": NaN, "personalized_closure_threshold": 0.25}')
        assert load_baseline_from_file(save_path) is None

        # 5. Non-existent path
        assert load_baseline_from_file(os.path.join(tmpdir, "missing.json")) is None


def test_websocket_calibration_control_commands():
    """Verify WebSocket command handling for calibration lifecycle."""
    import base64
    import cv2
    import numpy as np
    import time

    # Generate 64x64 valid frame
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", blank)
    valid_frame = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

    client = TestClient(app)
    with client.websocket_connect("/ws/detect") as ws:
        # 1. Send start_calibration command with valid frame
        ws.send_json({
            "command": "start_calibration",
            "driver_id": "ws_tester",
            "frame": valid_frame
        })
        resp = ws.receive_json()
        assert "calibration_state" in resp

        time.sleep(0.05)

        # 2. Send clear_baseline command
        ws.send_json({
            "command": "clear_baseline",
            "frame": valid_frame
        })
        resp2 = ws.receive_json()
        assert resp2.get("baseline_active") is False


def test_websocket_command_without_frame():
    """Verify WebSocket control commands succeed without needing a dummy video frame."""
    client = TestClient(app)
    with client.websocket_connect("/ws/detect") as ws:
        # Send start_calibration command without 'frame' field
        ws.send_json({
            "command": "start_calibration",
            "driver_id": "frame_free_tester"
        })
        resp = ws.receive_json()
        assert resp.get("status") == "success"
        assert resp.get("calibration_state") == "COLLECTING"
        assert "error" not in resp

        # Send clear_baseline command without 'frame' field
        ws.send_json({
            "command": "clear_baseline"
        })
        resp2 = ws.receive_json()
        assert resp2.get("status") == "success"
        assert resp2.get("baseline_active") is False
        assert "error" not in resp2


def test_camera_resolution_change_clears_personalized_baseline():
    """Verify camera resolution/stream change invalidates personalized baseline for safety."""
    detector = DrowsinessDetector("test_cam_change")
    # Apply valid custom baseline
    custom_baseline = DriverBaseline.default_fallback("driver_cam")
    custom_baseline.is_valid = True
    custom_baseline.personalized_closure_threshold = 0.22
    detector.apply_baseline(custom_baseline)
    assert detector.config.ear_threshold == 0.22
    assert detector.baseline.is_valid is True

    # Initial frame at 640x480
    res1 = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=OPEN_EYE_NORMAL,
        right_eye_points=OPEN_EYE_NORMAL,
        now=10.0,
        frame_width=640,
        frame_height=480
    )
    assert res1.baseline_active is True
    assert res1.calibrated_ear_threshold == 0.22

    # Second frame with same resolution preserves baseline
    res2 = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=OPEN_EYE_NORMAL,
        right_eye_points=OPEN_EYE_NORMAL,
        now=10.1,
        frame_width=640,
        frame_height=480
    )
    assert res2.baseline_active is True

    # Camera resolution changes to 1280x720 (different sensor / camera)
    res3 = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=OPEN_EYE_NORMAL,
        right_eye_points=OPEN_EYE_NORMAL,
        now=10.2,
        frame_width=1280,
        frame_height=720
    )
    # Baseline must be safely invalidated
    assert res3.baseline_active is False
    assert detector.baseline is None
    assert detector.config.ear_threshold == 0.25


def test_non_monotonic_timestamp_handling():
    """Verify non-monotonic or backwards clock frames are clamped and never yield negative durations."""
    detector = DrowsinessDetector("test_monotonic")
    res1 = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=CLOSED_EYE,
        right_eye_points=CLOSED_EYE,
        now=100.0
    )
    assert res1.timestamp == 100.0
    assert res1.closure_duration >= 0.0

    # Deliver frame with backward timestamp (clock jitter)
    res2 = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=CLOSED_EYE,
        right_eye_points=CLOSED_EYE,
        now=98.0
    )
    assert res2.timestamp == 100.0  # Clamped to previous monotonic time
    assert res2.closure_duration >= 0.0


def test_websocket_abort_calibration_lifecycle():
    """Verify WebSocket abort_calibration command sets state to FAILED and reason to USER_ABORTED."""
    client = TestClient(app)
    with client.websocket_connect("/ws/detect") as ws:
        # 1. Start calibration
        ws.send_json({
            "command": "start_calibration",
            "driver_id": "abort_tester",
            "session_id": "test_sid_abort_1"
        })
        resp1 = ws.receive_json()
        assert resp1.get("status") == "success"
        assert resp1.get("calibration_state") == "COLLECTING"
        assert resp1.get("client_session_id") == "test_sid_abort_1"

        # 2. Abort calibration
        ws.send_json({
            "command": "abort_calibration",
            "session_id": "test_sid_abort_2"
        })
        resp2 = ws.receive_json()
        assert resp2.get("status") == "success"
        assert resp2.get("calibration_state") == "FAILED"
        assert resp2.get("calibration_failure_reason") == "USER_ABORTED"
        assert resp2.get("client_session_id") == "test_sid_abort_2"


def test_websocket_repeated_start_calibration_handling():
    """Verify repeated start_calibration requests cleanly restart without state conflict."""
    client = TestClient(app)
    with client.websocket_connect("/ws/detect") as ws:
        # First start
        ws.send_json({
            "command": "start_calibration",
            "driver_id": "repeat_1"
        })
        resp1 = ws.receive_json()
        assert resp1.get("status") == "success"
        assert resp1.get("calibration_state") == "COLLECTING"

        # Second start immediately afterwards
        ws.send_json({
            "command": "start_calibration",
            "driver_id": "repeat_2"
        })
        resp2 = ws.receive_json()
        assert resp2.get("status") == "success"
        assert resp2.get("calibration_state") == "COLLECTING"
        assert "repeat_2" in resp2.get("status_message", "")


def test_websocket_client_session_id_stale_correlation():
    """Verify WebSocket echoes client_session_id allowing client to safely drop stale responses."""
    client = TestClient(app)
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_json({
            "command": "clear_baseline",
            "session_id": "frontend_session_alpha"
        })
        resp = ws.receive_json()
        assert resp.get("status") == "success"
        assert resp.get("client_session_id") == "frontend_session_alpha"
        assert resp.get("baseline_active") is False


def test_failed_calibration_clears_prior_baseline_safe_fallback():
    """Verify failed calibration never preserves a prior baseline, reverting to safe default (0.25)."""
    detector = DrowsinessDetector("test_fail_safety")

    # Apply prior valid baseline with lower threshold 0.22
    prior_baseline = DriverBaseline.default_fallback("driver_prior")
    prior_baseline.is_valid = True
    prior_baseline.personalized_closure_threshold = 0.22
    detector.apply_baseline(prior_baseline)
    assert detector.baseline is not None
    assert detector.config.ear_threshold == 0.22

    # Start new calibration
    detector.start_calibration("driver_prior")
    # Immediate check: starting calibration clears prior baseline for safety
    assert detector.baseline is None
    assert detector.config.ear_threshold == 0.25

    # Feed 50 frames open eyes (5.0s) then 35 frames closed eyes (3.5s prolonged closure -> suspected drowsiness)
    start_t = 5000.0
    for i in range(50):
        detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=OPEN_EYE_NORMAL,
            right_eye_points=OPEN_EYE_NORMAL,
            now=start_t + (i * 0.1)
        )
    for i in range(35):
        detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=CLOSED_EYE,
            right_eye_points=CLOSED_EYE,
            now=start_t + 5.0 + (i * 0.1)
        )

    # Finalize calibration
    failed_baseline = detector.finalize_calibration(now=start_t + 8.5)
    assert failed_baseline.is_valid is False
    assert failed_baseline.failure_reason == CalibrationFailureReason.SUSPECTED_DROWSINESS

    # Verify detector is NOT preserving prior baseline
    assert detector.baseline is None
    assert detector.config.ear_threshold == 0.25
    assert detector.calibrator.state == CalibrationState.FAILED
    assert detector.calibrator.failure_reason == CalibrationFailureReason.SUSPECTED_DROWSINESS

    # Next frame result surfaces the failure reason explicitly to the UI
    res = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=OPEN_EYE_NORMAL,
        right_eye_points=OPEN_EYE_NORMAL,
        now=start_t + 9.0
    )
    assert res.baseline_active is False
    assert res.calibrated_ear_threshold == 0.25
    assert res.calibration_state == "FAILED"
    assert res.calibration_failure_reason == "SUSPECTED_DROWSINESS"


def test_abort_calibration_clears_prior_baseline_safe_fallback():
    """Verify aborting calibration clears any prior baseline and enters FAILED/USER_ABORTED."""
    detector = DrowsinessDetector("test_abort_safety")

    # Prior baseline active
    prior_baseline = DriverBaseline.default_fallback("driver_prior")
    prior_baseline.is_valid = True
    prior_baseline.personalized_closure_threshold = 0.21
    detector.apply_baseline(prior_baseline)
    assert detector.baseline is not None

    # Start and then abort calibration
    detector.start_calibration("driver_prior")
    detector.abort_calibration()

    assert detector.baseline is None
    assert detector.config.ear_threshold == 0.25
    assert detector.calibrator.state == CalibrationState.FAILED
    assert detector.calibrator.failure_reason == CalibrationFailureReason.USER_ABORTED


def test_camera_resolution_change_during_active_calibration_aborts_collection():
    """Verify camera resolution change during active calibration aborts collection safely."""
    detector = DrowsinessDetector("test_cam_mid_calib")
    detector.start_calibration("driver_cam_mid")
    assert detector.calibrator.state == CalibrationState.COLLECTING

    # Frame 1: 640x480
    detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=OPEN_EYE_NORMAL,
        right_eye_points=OPEN_EYE_NORMAL,
        now=10.0,
        frame_width=640,
        frame_height=480
    )
    assert detector.calibrator.state == CalibrationState.COLLECTING

    # Frame 2: Resolution shifts to 1280x720 (camera switch or orientation change)
    detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=OPEN_EYE_NORMAL,
        right_eye_points=OPEN_EYE_NORMAL,
        now=10.1,
        frame_width=1280,
        frame_height=720
    )
    assert detector.calibrator.state == CalibrationState.FAILED
    assert detector.calibrator.failure_reason == CalibrationFailureReason.LOW_TRACKING_RATIO
    assert detector.baseline is None


def test_websocket_control_command_echoes_client_timestamp():
    """Verify control command responses echo client_timestamp to preserve ordering."""
    client = TestClient(app)
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_json({
            "command": "start_calibration",
            "driver_id": "ts_tester",
            "client_timestamp": 1700000000.5
        })
        resp = ws.receive_json()
        assert resp.get("status") == "success"
        assert resp.get("client_timestamp") == 1700000000.5


def test_retry_after_failed_calibration_succeeds():
    """Verify retrying calibration after a previous failure cleanly starts fresh and succeeds."""
    detector = DrowsinessDetector("test_retry_success")

    # Step 1: Failed calibration
    detector.start_calibration("driver_retry")
    start_t = 100.0
    for i in range(80):
        detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=CLOSED_EYE,
            right_eye_points=CLOSED_EYE,
            now=start_t + (i * 0.1)
        )
    b1 = detector.finalize_calibration(now=start_t + 8.0)
    assert b1.is_valid is False
    assert detector.calibrator.state == CalibrationState.FAILED

    # Step 2: Retry with fresh calibration and attentive open eyes
    calib_cfg = CalibrationConfig(
        min_duration_sec=8.0,
        target_duration_sec=10.0,
        min_valid_samples=50,
        closure_multiplier=0.70,
    )
    retry_start_t = 200.0
    detector.start_calibration("driver_retry", config=calib_cfg, start_timestamp=retry_start_t)
    assert detector.calibrator.state == CalibrationState.COLLECTING
    assert detector.calibrator.failure_reason == CalibrationFailureReason.NONE

    for i in range(100):
        detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=OPEN_EYE_NORMAL,
            right_eye_points=OPEN_EYE_NORMAL,
            now=retry_start_t + (i * 0.1)
        )

    b2 = detector.finalize_calibration(now=retry_start_t + 10.0)
    assert b2.is_valid is True
    assert detector.baseline is not None
    assert detector.baseline.is_valid is True
    assert detector.config.ear_threshold == b2.personalized_closure_threshold
    assert detector.config.ear_threshold != 0.25
