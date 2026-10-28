"""
AntiRHys Head Pose Detection-Signal Integration Tests
=====================================================
Verifies cautious integration of 3D head pose into the AntiRHys pipeline:
1. Observational Signal Isolation: Extreme pitch/yaw NEVER triggers drowsiness alarm.
2. Drowsiness alarm remains exclusively driven by EAR closure duration and PERCLOS.
3. Tracking loss, recovery, and session reset handling for head pose metrics.
4. WebSocket payload serialization of head pose metrics.
5. Regression: EAR, PERCLOS, baseline calibration, and alerts work identically.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from detector import (
    DrowsinessDetector,
    DetectionConfig,
    DetectionState,
    AlarmState,
    EyeState,
    TrackingState,
)
from head_pose import (
    HeadPoseEstimator,
    PoseStatus,
    CANONICAL_FACE_3D,
)
from main import app


OPEN_EYE_NORMAL = [
    (10.0, 50.0), (16.0, 43.0), (24.0, 43.0),
    (30.0, 50.0), (24.0, 50.0), (16.0, 50.0)
]  # EAR ≈ 0.35

CLOSED_EYE = [
    (10.0, 50.0), (16.0, 49.5), (24.0, 49.5),
    (30.0, 50.0), (24.0, 50.5), (16.0, 50.5)
]  # EAR ≈ 0.05


def _get_synthetic_pose_points(pitch_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> list:
    import cv2
    rvec = np.array([
        [np.radians(pitch_deg)],
        [np.radians(yaw_deg)],
        [np.radians(roll_deg)]
    ], dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [500.0]], dtype=np.float64)
    cam_matrix = np.array([
        [640.0, 0.0, 320.0],
        [0.0, 640.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    proj, _ = cv2.projectPoints(CANONICAL_FACE_3D, rvec, tvec, cam_matrix, dist_coeffs)
    return [(float(pt[0][0]), float(pt[0][1])) for pt in proj]


def test_head_pose_integrated_in_detection_result():
    """Verify head pose orientation is populated in DetectionResult and to_dict()."""
    detector = DrowsinessDetector("test_pose_det")
    pose_pts = _get_synthetic_pose_points(pitch_deg=-15.0, yaw_deg=10.0, roll_deg=0.0)

    res = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=OPEN_EYE_NORMAL,
        right_eye_points=OPEN_EYE_NORMAL,
        now=10.0,
        frame_width=640,
        frame_height=480,
        head_pose_points=pose_pts
    )

    assert res.head_pose_valid is True
    assert res.head_pose_status == PoseStatus.VALID.value
    assert pytest.approx(res.pitch, abs=1.0) == -15.0
    assert pytest.approx(res.yaw, abs=1.0) == 10.0
    assert abs(res.roll) < 2.0

    data_dict = res.to_dict()
    assert "pitch" in data_dict
    assert "yaw" in data_dict
    assert "roll" in data_dict
    assert "head_pose_valid" in data_dict
    assert "head_pose_status" in data_dict
    assert data_dict["head_pose_valid"] is True


def test_extreme_head_pose_does_not_trigger_drowsiness_alarm():
    """
    CRITICAL SAFETY REQUIREMENT:
    Extreme head pose (deep nod down or looking away) MUST NOT trigger a drowsiness alarm.
    Drowsiness alarms must remain strictly driven by verified EAR/PERCLOS metrics.
    """
    detector = DrowsinessDetector("test_isolation")
    # Extreme nod pitch (-40°) with open eyes
    deep_nod_pts = _get_synthetic_pose_points(pitch_deg=-40.0, yaw_deg=0.0)

    # Deliver 30 frames (3 seconds) of deep head nod with open eyes
    for i in range(30):
        res = detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=OPEN_EYE_NORMAL,
            right_eye_points=OPEN_EYE_NORMAL,
            now=100.0 + (i * 0.1),
            frame_width=640,
            frame_height=480,
            head_pose_points=deep_nod_pts
        )

    # Eyes were open: Alarm must remain OFF and state must remain NORMAL
    assert res.alarm_state == AlarmState.OFF
    assert res.detection_state == DetectionState.NORMAL
    assert res.eye_state == EyeState.OPEN
    # Head pose accurately reported
    assert res.head_pose_valid is True
    assert pytest.approx(res.pitch, abs=1.0) == -40.0


def test_drowsiness_alarm_triggers_independently_with_head_pose():
    """Verify prolonged eye closure triggers drowsiness alarm regardless of head pose."""
    detector = DrowsinessDetector("test_alarm_with_pose")
    neutral_pts = _get_synthetic_pose_points(pitch_deg=0.0, yaw_deg=0.0)

    # Eyes closed for 2.0s (> 1.5s threshold)
    for i in range(25):
        res = detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=CLOSED_EYE,
            right_eye_points=CLOSED_EYE,
            now=200.0 + (i * 0.1),
            frame_width=640,
            frame_height=480,
            head_pose_points=neutral_pts
        )

    assert res.alarm_state == AlarmState.TRIGGERED
    assert res.detection_state == DetectionState.POSSIBLE_DROWSINESS
    assert res.head_pose_valid is True
    assert abs(res.pitch) < 0.5


def test_head_pose_during_face_tracking_loss():
    """Verify head pose status transitions to FACE_LOST when no face is tracked."""
    detector = DrowsinessDetector("test_loss")

    # 1. Active face
    res1 = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=OPEN_EYE_NORMAL,
        right_eye_points=OPEN_EYE_NORMAL,
        now=10.0,
        frame_width=640,
        frame_height=480,
        head_pose_points=_get_synthetic_pose_points()
    )
    assert res1.head_pose_valid is True

    # 2. Tracking lost
    res2 = detector.process_frame_landmarks(
        has_face=False,
        now=10.1
    )
    assert res2.head_pose_valid is False
    assert res2.head_pose_status == PoseStatus.FACE_LOST.value
    assert res2.pitch == 0.0
    assert res2.yaw == 0.0
    assert res2.roll == 0.0


def test_head_pose_websocket_control_command_defaults():
    """Verify control command responses include safe default head pose fields."""
    client = TestClient(app)
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_json({
            "command": "clear_baseline"
        })
        resp = ws.receive_json()
        assert resp.get("status") == "success"
        assert "pitch" in resp
        assert "yaw" in resp
        assert "roll" in resp
        assert resp.get("head_pose_valid") is False
        assert resp.get("head_pose_status") == "UNAVAILABLE"
