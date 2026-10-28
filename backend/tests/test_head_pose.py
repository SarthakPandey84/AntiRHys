"""
AntiRHys 3D Head Pose Estimator Unit Tests
==========================================
Deterministic tests verifying Perspective-n-Point head pose calculations:
1. Canonical synthetic projections (neutral, pitch nod, yaw turn, roll tilt).
2. Angle conventions and sign correctness.
3. Pre-validation: Missing, none, empty, invalid point count, NaN, Inf coordinates.
4. Degenerate geometries: zero-spread, single-point, collinear points.
5. Reprojection error calculation and outlier rejection.
6. Deterministic stability for repeated identical inputs.
7. Sensor dimension edge cases (zero/negative frame width/height).
"""

import math
import cv2
import numpy as np
import pytest

from head_pose import (
    HeadPoseEstimator,
    HeadPoseResult,
    PoseStatus,
    CANONICAL_FACE_3D,
    HEAD_POSE_LANDMARK_INDICES,
)


def _generate_synthetic_2d(
    rvec: np.ndarray,
    tvec: np.ndarray = np.array([[0.0], [0.0], [500.0]]),
    width: int = 640,
    height: int = 480
) -> list:
    """Project canonical 3D face model points into 2D pixel coordinates with known rotation."""
    cam_matrix = np.array([
        [float(width), 0.0, float(width) / 2.0],
        [0.0, float(width), float(height) / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    proj, _ = cv2.projectPoints(CANONICAL_FACE_3D, rvec, tvec, cam_matrix, dist_coeffs)
    return [(float(pt[0][0]), float(pt[0][1])) for pt in proj]


def test_head_pose_neutral_forward():
    """Verify neutral forward-facing head returns ~0.0 deg for pitch, yaw, and roll."""
    estimator = HeadPoseEstimator()
    rvec_neutral = np.zeros((3, 1), dtype=np.float64)
    points_2d = _generate_synthetic_2d(rvec_neutral)

    result = estimator.estimate_pose(points_2d, 640, 480)
    assert result.is_valid is True
    assert result.status == PoseStatus.VALID.value
    assert abs(result.pitch) < 0.5
    assert abs(result.yaw) < 0.5
    assert abs(result.roll) < 0.5
    assert result.reprojection_error < 0.1
    assert result.confidence > 0.95


def test_head_pose_pitch_nod_down_convention():
    """
    Verify pitch angle convention:
    Chin dropped down (micro-sleep nod) must produce a negative pitch angle (< 0°).
    """
    estimator = HeadPoseEstimator()
    nod_deg = -20.0
    rvec_nod = np.array([[np.radians(nod_deg)], [0.0], [0.0]], dtype=np.float64)
    points_2d = _generate_synthetic_2d(rvec_nod)

    result = estimator.estimate_pose(points_2d, 640, 480)
    assert result.is_valid is True
    assert pytest.approx(result.pitch, abs=0.5) == nod_deg
    assert abs(result.yaw) < 0.5
    assert abs(result.roll) < 0.5


def test_head_pose_pitch_look_up_convention():
    """
    Verify pitch angle convention:
    Head tilted upward must produce a positive pitch angle (> 0°).
    """
    estimator = HeadPoseEstimator()
    up_deg = 15.0
    rvec_up = np.array([[np.radians(up_deg)], [0.0], [0.0]], dtype=np.float64)
    points_2d = _generate_synthetic_2d(rvec_up)

    result = estimator.estimate_pose(points_2d, 640, 480)
    assert result.is_valid is True
    assert pytest.approx(result.pitch, abs=0.5) == up_deg
    assert abs(result.yaw) < 0.5
    assert abs(result.roll) < 0.5


def test_head_pose_yaw_turn_conventions():
    """
    Verify yaw angle conventions:
    Negative yaw = driver right (camera left).
    Positive yaw = driver left (camera right).
    """
    estimator = HeadPoseEstimator()

    # Yaw right (-25 deg)
    rvec_right = np.array([[0.0], [np.radians(-25.0)], [0.0]], dtype=np.float64)
    res_right = estimator.estimate_pose(_generate_synthetic_2d(rvec_right), 640, 480)
    assert res_right.is_valid is True
    assert pytest.approx(res_right.yaw, abs=0.5) == -25.0

    # Yaw left (+25 deg)
    rvec_left = np.array([[0.0], [np.radians(25.0)], [0.0]], dtype=np.float64)
    res_left = estimator.estimate_pose(_generate_synthetic_2d(rvec_left), 640, 480)
    assert res_left.is_valid is True
    assert pytest.approx(res_left.yaw, abs=0.5) == 25.0


def test_head_pose_roll_tilt_conventions():
    """Verify roll angle convention for clockwise and counter-clockwise head tilt."""
    estimator = HeadPoseEstimator()

    # Tilt clockwise (+15 deg)
    rvec_cw = np.array([[0.0], [0.0], [np.radians(15.0)]], dtype=np.float64)
    res_cw = estimator.estimate_pose(_generate_synthetic_2d(rvec_cw), 640, 480)
    assert res_cw.is_valid is True
    assert pytest.approx(res_cw.roll, abs=0.5) == 15.0

    # Tilt counter-clockwise (-15 deg)
    rvec_ccw = np.array([[0.0], [0.0], [np.radians(-15.0)]], dtype=np.float64)
    res_ccw = estimator.estimate_pose(_generate_synthetic_2d(rvec_ccw), 640, 480)
    assert res_ccw.is_valid is True
    assert pytest.approx(res_ccw.roll, abs=0.5) == -15.0


def test_head_pose_missing_or_empty_landmarks():
    """Verify estimator handles None or empty input safely without exceptions."""
    estimator = HeadPoseEstimator()
    res1 = estimator.estimate_pose(None, 640, 480)
    assert res1.is_valid is False
    assert res1.status == PoseStatus.FACE_LOST.value

    res2 = estimator.estimate_pose([], 640, 480)
    assert res2.is_valid is False
    assert res2.status == PoseStatus.FACE_LOST.value


def test_head_pose_invalid_point_count():
    """Verify estimator rejects landmark inputs with count != 6."""
    estimator = HeadPoseEstimator()
    pts_3 = [(100.0, 100.0), (200.0, 200.0), (300.0, 300.0)]
    res = estimator.estimate_pose(pts_3, 640, 480)
    assert res.is_valid is False
    assert res.status == PoseStatus.INVALID_POINT_COUNT.value


def test_head_pose_non_finite_coordinates():
    """Verify estimator rejects NaN and Inf coordinates without raising OpenCV exceptions."""
    estimator = HeadPoseEstimator()
    pts_nan = [(float("nan"), 100.0)] + [(100.0, 100.0)] * 5
    res_nan = estimator.estimate_pose(pts_nan, 640, 480)
    assert res_nan.is_valid is False
    assert res_nan.status == PoseStatus.NON_FINITE_COORDINATES.value

    pts_inf = [(100.0, float("inf"))] + [(100.0, 100.0)] * 5
    res_inf = estimator.estimate_pose(pts_inf, 640, 480)
    assert res_inf.is_valid is False
    assert res_inf.status == PoseStatus.NON_FINITE_COORDINATES.value


def test_head_pose_degenerate_geometries():
    """Verify estimator rejects collapsed or collinear points as DEGENERATE_GEOMETRY."""
    estimator = HeadPoseEstimator()

    # All points identical at (0, 0)
    pts_zero = [(0.0, 0.0)] * 6
    res_zero = estimator.estimate_pose(pts_zero, 640, 480)
    assert res_zero.is_valid is False
    assert res_zero.status == PoseStatus.DEGENERATE_GEOMETRY.value

    # Horizontal line (span_h == 0)
    pts_collinear = [(10.0 * i, 100.0) for i in range(6)]
    res_line = estimator.estimate_pose(pts_collinear, 640, 480)
    assert res_line.is_valid is False
    assert res_line.status == PoseStatus.DEGENERATE_GEOMETRY.value


def test_head_pose_invalid_frame_dimensions():
    """Verify estimator handles zero or negative frame dimensions safely."""
    estimator = HeadPoseEstimator()
    pts = [(100.0, 100.0)] * 6
    res = estimator.estimate_pose(pts, 0, 480)
    assert res.is_valid is False
    assert res.status == PoseStatus.UNAVAILABLE.value


def test_head_pose_reprojection_error_gating():
    """Verify scrambled/outlier landmarks that violate facial geometry are rejected."""
    estimator = HeadPoseEstimator(max_reprojection_error=10.0)
    # Scrambled coordinates with huge distortion
    scrambled = [
        (320.0, 240.0),  # Nose
        (320.0, 50.0),   # Chin placed ABOVE nose!
        (50.0, 400.0),   # Left eye placed below mouth
        (600.0, 400.0),  # Right eye placed below mouth
        (320.0, 100.0),
        (320.0, 300.0),
    ]
    res = estimator.estimate_pose(scrambled, 640, 480)
    assert res.is_valid is False
    assert res.status in (PoseStatus.REPROJECTION_ERROR_HIGH.value, PoseStatus.SOLVE_FAILED.value)


def test_head_pose_deterministic_stability():
    """Verify repeated calls on identical landmark inputs produce bit-identical outputs."""
    estimator = HeadPoseEstimator()
    rvec = np.array([[-np.radians(12.5)], [np.radians(8.2)], [np.radians(-3.1)]], dtype=np.float64)
    pts = _generate_synthetic_2d(rvec)

    res_first = estimator.estimate_pose(pts, 640, 480)
    for _ in range(50):
        res = estimator.estimate_pose(pts, 640, 480)
        assert res.pitch == res_first.pitch
        assert res.yaw == res_first.yaw
        assert res.roll == res_first.roll
        assert res.reprojection_error == res_first.reprojection_error
        assert res.confidence == res_first.confidence
