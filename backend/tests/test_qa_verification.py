"""
Comprehensive Senior QA Verification Suite for AntiRHys.
Tests Eye Aspect Ratio, PerclosCalculator, DrowsinessDetector,
State Machine, and Backend WebSocket/API.
"""

import base64
import csv
import json
import math
import os
import time
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect
from fastapi.testclient import TestClient

import main
from main import app
from detector import (
    calculate_ear,
    DrowsinessDetector,
    DetectionConfig,
    DetectionState,
    AlarmState,
    TrackingState,
    EyeState,
    PerclosCalculator,
    PerclosStatus
)
from session_logger import SessionLogger

OPEN_EYE = [
    (10.0, 50.0), (16.0, 43.0), (24.0, 43.0),
    (30.0, 50.0), (24.0, 50.0), (16.0, 50.0)
]
CLOSED_EYE = [
    (10.0, 50.0), (16.0, 49.5), (24.0, 49.5),
    (30.0, 50.0), (24.0, 50.5), (16.0, 50.5)
]


# ============================================================================
# PHASE 2 - A. Eye Aspect Ratio (EAR) Verification
# ============================================================================

def test_ear_open_eyes():
    ear = calculate_ear(OPEN_EYE)
    assert 0.30 <= ear <= 0.40


def test_ear_closed_eyes():
    ear = calculate_ear(CLOSED_EYE)
    assert 0.01 <= ear <= 0.15


def test_ear_zero_width_and_degenerate():
    # Coincident p1 and p4
    pts = [(50.0, 50.0)] * 6
    assert calculate_ear(pts, min_width=1.0) == 0.0

    # Width less than min_width (0.5 < 1.0)
    narrow = [
        (10.0, 50.0), (10.2, 45.0), (10.3, 45.0),
        (10.5, 50.0), (10.3, 50.0), (10.2, 50.0)
    ]
    assert calculate_ear(narrow, min_width=1.0) == 0.0


def test_ear_missing_landmarks():
    assert calculate_ear([]) == 0.0
    assert calculate_ear(OPEN_EYE[:5]) == 0.0
    assert calculate_ear(OPEN_EYE + [(35.0, 50.0)]) == 0.0


def test_ear_invalid_and_non_finite_coordinates():
    # NaN in lateral point
    bad_nan = OPEN_EYE.copy()
    bad_nan[0] = (float('nan'), 50.0)
    assert calculate_ear(bad_nan) == 0.0

    # Inf in vertical point
    bad_inf = OPEN_EYE.copy()
    bad_inf[2] = (24.0, float('inf'))
    assert calculate_ear(bad_inf) == 0.0

    # -Inf in medial point
    bad_ninf = OPEN_EYE.copy()
    bad_ninf[3] = (float('-inf'), 50.0)
    assert calculate_ear(bad_ninf) == 0.0


# ============================================================================
# PHASE 2 - B. PERCLOS Unit & Algorithm Verification
# ============================================================================

def test_perclos_eyes_open_throughout_window():
    calc = PerclosCalculator(window_duration=60.0, min_observation_duration=20.0, closure_ear_threshold=0.25)
    t = 0.0
    for _ in range(600):  # 60s at 10 Hz
        calc.record_observation(ear=0.35, now=t)
        t += 0.1

    perclos, is_valid, status, total_valid, total_closed = calc.compute_perclos(t)
    assert perclos == pytest.approx(0.0, abs=1e-3)
    assert is_valid is True
    assert status == PerclosStatus.VALID.value
    assert total_valid == pytest.approx(60.0, abs=0.5)
    assert total_closed == pytest.approx(0.0, abs=1e-3)


def test_perclos_eyes_closed_throughout_window():
    calc = PerclosCalculator(window_duration=60.0, min_observation_duration=20.0, closure_ear_threshold=0.25)
    t = 0.0
    for _ in range(600):  # 60s at 10 Hz
        calc.record_observation(ear=0.05, now=t)
        t += 0.1

    perclos, is_valid, status, total_valid, total_closed = calc.compute_perclos(t)
    assert perclos == pytest.approx(1.0, abs=1e-3)
    assert is_valid is True
    assert status == PerclosStatus.VALID.value
    assert total_valid == pytest.approx(60.0, abs=0.5)
    assert total_closed == pytest.approx(60.0, abs=0.5)


def test_perclos_natural_intermittent_blinking():
    # 60 seconds with 0.2s blinks every 4 seconds (15 blinks = 3.0s total closed time)
    calc = PerclosCalculator(window_duration=60.0, min_observation_duration=20.0, closure_ear_threshold=0.25)
    t = 0.0
    expected_closed = 0.0
    for _ in range(600):
        in_blink = (t % 4.0) < 0.20
        ear = 0.05 if in_blink else 0.35
        calc.record_observation(ear=ear, now=t)
        if in_blink:
            expected_closed += 0.1
        t += 0.1

    perclos, is_valid, status, total_valid, total_closed = calc.compute_perclos(t)
    assert is_valid is True
    assert status == PerclosStatus.VALID.value
    # 3.0s closed out of 60.0s valid observation = ~5% (0.05)
    assert perclos == pytest.approx(3.0 / 60.0, abs=0.015)
    assert total_closed == pytest.approx(3.0, abs=0.5)


def test_perclos_rolling_window_expiration():
    # First 30s eyes closed. Next 60s eyes open.
    # At t=90s, window is [30s, 90s]. All closed eye time should have expired.
    calc = PerclosCalculator(window_duration=60.0, min_observation_duration=20.0, closure_ear_threshold=0.25)
    t = 0.0
    # 0 to 30s closed
    for _ in range(300):
        calc.record_observation(ear=0.05, now=t)
        t += 0.1
    p30, valid30, _, _, _ = calc.compute_perclos(t)
    assert p30 == pytest.approx(1.0, abs=1e-3)
    assert valid30 is True

    # 30 to 90s open (60s)
    for _ in range(600):
        calc.record_observation(ear=0.35, now=t)
        t += 0.1

    p90, valid90, status90, total_valid90, total_closed90 = calc.compute_perclos(t)
    assert p90 == pytest.approx(0.0, abs=1e-3)
    assert valid90 is True
    assert status90 == PerclosStatus.VALID.value
    assert total_closed90 == pytest.approx(0.0, abs=1e-3)
    assert total_valid90 == pytest.approx(60.0, abs=0.5)


def test_perclos_startup_insufficient_valid_observations():
    # Only 10s of data in a 60s window with min_observation=20s
    calc = PerclosCalculator(window_duration=60.0, min_observation_duration=20.0, closure_ear_threshold=0.25)
    t = 100.0
    for _ in range(100):  # 10s
        calc.record_observation(ear=0.35, now=t)
        t += 0.1

    perclos, is_valid, status, total_valid, _ = calc.compute_perclos(t)
    assert is_valid is False
    assert status == PerclosStatus.CALCULATING.value
    assert total_valid == pytest.approx(10.0, abs=0.5)


def test_perclos_irregular_frame_intervals_and_time_based_integration():
    # Frames arriving at irregular intervals (e.g. 50ms, 150ms, 250ms)
    calc = PerclosCalculator(window_duration=60.0, min_observation_duration=20.0, closure_ear_threshold=0.25)
    t = 0.0
    intervals = [0.05, 0.15, 0.20, 0.10]
    idx = 0
    while t < 30.0:
        dt = intervals[idx % len(intervals)]
        calc.record_observation(ear=0.35, now=t)
        t += dt
        idx += 1

    perclos, is_valid, status, total_valid, total_closed = calc.compute_perclos(t)
    assert is_valid is True
    assert status == PerclosStatus.VALID.value
    assert total_valid == pytest.approx(30.0, abs=1.0)
    assert total_closed == 0.0


def test_perclos_duplicate_stale_or_out_of_order_timestamps():
    calc = PerclosCalculator(window_duration=60.0, min_observation_duration=20.0)
    calc.record_observation(ear=0.35, now=10.0)
    # Duplicate timestamp
    calc.record_observation(ear=0.35, now=10.0)
    # Stale/out-of-order timestamp
    calc.record_observation(ear=0.35, now=9.5)
    calc.record_observation(ear=0.35, now=10.1)

    # Must not raise exceptions or crash
    perclos, is_valid, status, total_valid, _ = calc.compute_perclos(10.1)
    assert math.isfinite(perclos)


def test_perclos_tracking_loss_and_recovery():
    # 20s open, then 10s tracking lost, then 20s open
    calc = PerclosCalculator(window_duration=60.0, min_observation_duration=20.0)
    t = 0.0
    for _ in range(200):  # 20s
        calc.record_observation(0.35, t)
        t += 0.1

    # Tracking lost for 10s: call record_invalid_observation
    calc.record_invalid_observation()
    t += 10.0

    # Tracking recovered: 20s open
    for _ in range(200):
        calc.record_observation(0.35, t)
        t += 0.1

    perclos, is_valid, status, total_valid, total_closed = calc.compute_perclos(t)
    # Total valid time must be ~40s (the 10s gap must NOT be credited as valid time)
    assert total_valid == pytest.approx(40.0, abs=1.0)
    assert is_valid is True
    assert status == PerclosStatus.VALID.value


def test_perclos_session_reset():
    calc = PerclosCalculator(window_duration=60.0, min_observation_duration=20.0)
    t = 0.0
    for _ in range(300):
        calc.record_observation(0.05, t)
        t += 0.1
    p, is_valid, _, _, _ = calc.compute_perclos(t)
    assert is_valid is True

    calc.reset()
    p_after, is_valid_after, status_after, valid_sec, closed_sec = calc.compute_perclos(t)
    assert p_after == 0.0
    assert is_valid_after is False
    assert status_after in (PerclosStatus.CALCULATING.value, PerclosStatus.INSUFFICIENT_DATA.value)
    assert valid_sec == 0.0
    assert closed_sec == 0.0


# ============================================================================
# PHASE 2 - B & C. DrowsinessDetector Integration of PERCLOS & State Machine
# ============================================================================

def test_detector_perclos_connected_in_results():
    """
    Verifies whether PERCLOS is actually computed and returned by DrowsinessDetector.
    """
    detector = DrowsinessDetector(session_id="test_perclos_sess")
    t = 100.0
    # Process 250 frames (25s @ 10 Hz) with closed eyes
    for _ in range(250):
        res = detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=CLOSED_EYE,
            right_eye_points=CLOSED_EYE,
            now=t
        )
        t += 0.1

    # On current code, this test will expose if PERCLOS is connected or disconnected
    assert res.perclos_valid is True, f"Expected perclos_valid to be True, got {res.perclos_valid} (status: {res.perclos_status})"
    assert res.perclos > 0.80, f"Expected perclos > 0.80 for continuous closed eyes, got {res.perclos}"
    assert res.perclos_status == "VALID"


def test_detector_perclos_reset_clears_metrics():
    """
    Verifies that detector.reset_session() resets PERCLOS history.
    """
    detector = DrowsinessDetector(session_id="sess_1")
    t = 100.0
    for _ in range(250):
        detector.process_frame_landmarks(True, CLOSED_EYE, CLOSED_EYE, now=t)
        t += 0.1

    detector.reset_session("sess_2")
    res = detector.process_frame_landmarks(True, OPEN_EYE, OPEN_EYE, now=t)
    assert res.perclos_valid is False
    assert res.perclos_status in ("CALCULATING", "INSUFFICIENT_DATA")


def test_detector_tracking_loss_does_not_credit_perclos():
    """
    Verifies that tracking loss does not credit valid observation time to PERCLOS.
    """
    detector = DrowsinessDetector(session_id="sess_track")
    t = 100.0
    # 5s open
    for _ in range(50):
        detector.process_frame_landmarks(True, OPEN_EYE, OPEN_EYE, now=t)
        t += 0.1

    # 15s tracking lost
    for _ in range(150):
        detector.process_frame_landmarks(False, now=t)
        t += 0.1

    res = detector.process_frame_landmarks(True, OPEN_EYE, OPEN_EYE, now=t)
    # Only ~5.1s of valid observation, so must NOT be VALID yet
    assert res.perclos_valid is False
    assert res.perclos_status == "CALCULATING"


# ============================================================================
# PHASE 2 - D. Backend API & WebSocket Edge Cases
# ============================================================================

@pytest.fixture
def client():
    return TestClient(app)


def test_api_invalid_image_dimensions(client):
    """
    Test frames with dimensions below minimum (64x64).
    """
    tiny_img = np.zeros((32, 32, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', tiny_img)
    b64 = base64.b64encode(buffer).decode('utf-8')
    payload = f"data:image/jpeg;base64,{b64}"

    with client.websocket_connect("/ws/detect") as ws:
        ws.send_text(json.dumps({"frame": payload}))
        res = ws.receive_json()
        assert "error" in res
        assert "resolution too small" in res["error"].lower()


def test_api_logging_failure_resilience(client):
    """
    Verify that if the session logger encounters an OS write failure,
    the detection websocket connection does NOT crash or terminate.
    """
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', blank)
    b64 = base64.b64encode(buf).decode('utf-8')
    payload = f"data:image/jpeg;base64,{b64}"

    with patch.object(main.session_logger, 'log_event', side_effect=OSError("Disk full simulation")):
        with client.websocket_connect("/ws/detect") as ws:
            ws.send_text(json.dumps({"frame": payload}))
            # Must still respond gracefully without disconnecting
            res = ws.receive_json()
            assert "session_id" in res


def test_api_max_concurrent_sessions_limit(client):
    """
    Verify MAX_CONCURRENT_SESSIONS ceiling closes excess connections with 1013 code.
    """
    orig_max = main.MAX_CONCURRENT_SESSIONS
    try:
        main.MAX_CONCURRENT_SESSIONS = 1
        with client.websocket_connect("/ws/detect") as ws1:
            # Second concurrent connection should be rejected
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect("/ws/detect"):
                    pass
            assert exc.value.code == 1013
    finally:
        main.MAX_CONCURRENT_SESSIONS = orig_max


# ============================================================================
# PHASE 2 - E. Configuration, Security & Robustness Tests
# ============================================================================

def test_detection_config_defaults_and_env_override():
    """Verify DetectionConfig loads defaults and safely overrides from environment."""
    cfg = DetectionConfig()
    assert cfg.ear_threshold == 0.25
    assert cfg.closure_duration_threshold == 1.5
    assert cfg.perclos_warning_threshold == 0.20
    assert cfg.perclos_window_sec == 60.0

    # Test env override
    with patch.dict(os.environ, {
        "EAR_THRESHOLD": "0.24",
        "CLOSURE_DURATION_THRESHOLD": "2.0",
        "PERCLOS_WARNING_THRESHOLD": "0.25",
        "PERCLOS_WINDOW_SEC": "45.0",
        "PERCLOS_MIN_OBSERVATION_SEC": "15.0"
    }):
        env_cfg = DetectionConfig.from_env()
        assert env_cfg.ear_threshold == 0.24
        assert env_cfg.closure_duration_threshold == 2.0
        assert env_cfg.perclos_warning_threshold == 0.25
        assert env_cfg.perclos_window_sec == 45.0
        assert env_cfg.perclos_min_observation_sec == 15.0

    # Test invalid env override fallbacks safely
    with patch.dict(os.environ, {"EAR_THRESHOLD": "not-a-number"}):
        fallback_cfg = DetectionConfig.from_env()
        assert fallback_cfg.ear_threshold == 0.25


def test_detector_process_frame_error_state():
    """Verify process_frame_error transitions detector to CAMERA_OR_PROCESSING_ERROR and resets closure timing."""
    detector = DrowsinessDetector(session_id="err_sess")
    # Simulate partial closure
    detector.process_frame_landmarks(True, CLOSED_EYE, CLOSED_EYE, now=10.0)
    assert detector.closure_start_time is not None

    # Error occurs
    res = detector.process_frame_error("Simulated camera decode failure")
    assert res.detection_state == DetectionState.CAMERA_OR_PROCESSING_ERROR
    assert "Simulated camera decode failure" in res.status_message
    assert detector.closure_start_time is None
    assert detector.closure_duration == 0.0


def test_cors_origin_case_insensitivity():
    """Verify origin validation is case-insensitive and normalizes trailing slashes."""
    with patch.object(main, 'ALLOWED_ORIGINS', ["http://localhost:5173", "https://antirhys.app"]):
        assert main.is_origin_allowed("http://localhost:5173") is True
        assert main.is_origin_allowed("HTTP://LOCALHOST:5173/") is True
        assert main.is_origin_allowed("https://antirhys.app") is True
        assert main.is_origin_allowed("https://evil-site.com") is False


def test_session_logger_float_sanitization(tmp_path):
    """Verify SessionLogger sanitizes non-finite floats (NaN, Inf) without throwing."""
    log_file = tmp_path / "test_session_log.csv"
    logger = SessionLogger(str(log_file))

    # Log non-finite numbers
    logger.log_event(
        session_id="sanitized_sess",
        event_type="TEST",
        detection_state="NORMAL",
        ear=float("nan"),
        closure_duration=float("inf"),
        perclos=-float("inf")
    )

    # Read back and verify CSV
    with open(str(log_file), mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["ear"] == "0.000"
        assert rows[0]["closure_duration"] == "0.00"
        assert rows[0]["perclos"] == "0.000"


def test_api_invalid_base64_returns_error_state(client):
    """Verify sending malformed base64 returns CAMERA_OR_PROCESSING_ERROR detection state."""
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_text(json.dumps({"frame": "data:image/jpeg;base64,!!!NOT_VALID_BASE64!!!"}))
        res = ws.receive_json()
        assert "error" in res
        assert "Invalid base64 encoding" in res["error"]
        assert res.get("detection_state") == "CAMERA_OR_PROCESSING_ERROR"
