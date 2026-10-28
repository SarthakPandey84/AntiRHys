"""
Unit tests for Personalized Driver Baseline Calibrator (AntiRHys Phase 5)
========================================================================
Validates deterministic behavior of calibration gating, tracking quality checks,
drowsiness rejection, threshold bounds clamping, and dictionary serialization.
"""

import pytest
import time
from baseline_calibrator import (
    BaselineCalibrator,
    CalibrationConfig,
    CalibrationState,
    CalibrationFailureReason,
    DriverBaseline,
)


def test_successful_calibration():
    """Verify valid calibration with typical attentive driver distribution."""
    config = CalibrationConfig(
        min_duration_sec=8.0,
        target_duration_sec=10.0,
        min_valid_samples=50,
        min_tracking_ratio=0.80,
        closure_multiplier=0.70,
        min_closure_threshold=0.18,
        max_closure_threshold=0.32,
    )
    calibrator = BaselineCalibrator(config=config)
    start_t = 100.0
    calibrator.start_calibration(driver_id="driver_001", start_timestamp=start_t)

    # Feed 10 seconds of 10 FPS attentive frames with normal open-eye EAR ~ 0.35
    for i in range(100):
        t = start_t + (i * 0.1)
        # 2 normal blinks at frames 30 and 70 (0.1s each)
        if i in (30, 70):
            ear = 0.12
        else:
            ear = 0.35 + (0.01 if i % 2 == 0 else -0.01)
        calibrator.add_sample(timestamp=t, ear=ear, face_detected=True, is_valid_geometry=True)

    end_t = start_t + 10.0
    baseline = calibrator.finalize(end_timestamp=end_t)

    assert calibrator.state == CalibrationState.COMPLETED
    assert baseline.is_valid is True
    assert baseline.failure_reason == CalibrationFailureReason.NONE
    assert baseline.driver_id == "driver_001"
    assert baseline.sample_count == 100
    assert baseline.tracking_ratio == 1.0
    assert 0.33 <= baseline.median_open_ear <= 0.37
    # 70% of ~0.35 = ~0.245, well within [0.18, 0.32]
    assert 0.23 <= baseline.personalized_closure_threshold <= 0.26


def test_rejection_insufficient_duration():
    """Verify failure when calibration period is terminated prematurely."""
    calibrator = BaselineCalibrator()
    start_t = 100.0
    calibrator.start_calibration(start_timestamp=start_t)

    # Feed 5 seconds (below min 8.0s)
    for i in range(60):
        calibrator.add_sample(
            timestamp=start_t + (i * 0.08),
            ear=0.32,
            face_detected=True,
            is_valid_geometry=True
        )

    baseline = calibrator.finalize(end_timestamp=start_t + 4.8)
    assert calibrator.state == CalibrationState.FAILED
    assert baseline.is_valid is False
    assert baseline.failure_reason == CalibrationFailureReason.INSUFFICIENT_DURATION
    assert baseline.personalized_closure_threshold == calibrator.config.fallback_closure_threshold


def test_rejection_low_tracking_ratio():
    """Verify failure when face tracking drops below 80% of frames."""
    calibrator = BaselineCalibrator()
    start_t = 100.0
    calibrator.start_calibration(start_timestamp=start_t)

    # 100 frames over 10 seconds, but 35 frames have face_detected=False
    for i in range(100):
        tracked = i >= 35  # only 65% tracked
        calibrator.add_sample(
            timestamp=start_t + (i * 0.1),
            ear=0.32 if tracked else 0.0,
            face_detected=tracked,
            is_valid_geometry=tracked
        )

    baseline = calibrator.finalize(end_timestamp=start_t + 10.0)
    assert calibrator.state == CalibrationState.FAILED
    assert baseline.is_valid is False
    assert baseline.failure_reason == CalibrationFailureReason.LOW_TRACKING_RATIO


def test_rejection_suspected_drowsiness_prolonged_closure():
    """Verify calibration is aborted if driver displays prolonged eye closure during calibration."""
    calibrator = BaselineCalibrator()
    start_t = 100.0
    calibrator.start_calibration(start_timestamp=start_t)

    # 10 seconds: normal for 4s, closed for 1.2s (> max 0.75s), normal for 4.8s
    for i in range(100):
        t = start_t + (i * 0.1)
        if 40 <= i <= 52:  # 1.3 seconds closed
            ear = 0.10
        else:
            ear = 0.34
        calibrator.add_sample(timestamp=t, ear=ear, face_detected=True, is_valid_geometry=True)

    baseline = calibrator.finalize(end_timestamp=start_t + 10.0)
    assert calibrator.state == CalibrationState.FAILED
    assert baseline.is_valid is False
    assert baseline.failure_reason == CalibrationFailureReason.SUSPECTED_DROWSINESS


def test_clamping_bounds():
    """Verify that personalized thresholds are strictly clamped between 0.18 and 0.32."""
    config = CalibrationConfig(
        closure_multiplier=0.70,
        min_closure_threshold=0.18,
        max_closure_threshold=0.32,
    )

    # Case A: Driver with exceptionally high open EAR (e.g. 0.49 -> 0.49 * 0.70 = 0.343, clamped to 0.32)
    calibrator_high = BaselineCalibrator(config=config)
    start_t = 100.0
    calibrator_high.start_calibration(start_timestamp=start_t)
    for i in range(100):
        calibrator_high.add_sample(timestamp=start_t + (i * 0.1), ear=0.48, face_detected=True)
    baseline_high = calibrator_high.finalize(end_timestamp=start_t + 10.0)
    assert baseline_high.is_valid is True
    assert baseline_high.personalized_closure_threshold == 0.32

    # Case B: Driver with naturally narrower open EAR (e.g. 0.23 -> 0.23 * 0.70 = 0.161, clamped to 0.18)
    calibrator_low = BaselineCalibrator(config=config)
    calibrator_low.start_calibration(start_timestamp=start_t)
    for i in range(100):
        calibrator_low.add_sample(timestamp=start_t + (i * 0.1), ear=0.23, face_detected=True)
    baseline_low = calibrator_low.finalize(end_timestamp=start_t + 10.0)
    assert baseline_low.is_valid is True
    assert baseline_low.personalized_closure_threshold == 0.18


def test_serialization_and_reset():
    """Verify dictionary serialization/deserialization and reset behavior."""
    baseline = DriverBaseline.default_fallback(driver_id="user_test")
    d = baseline.to_dict()
    assert d["driver_id"] == "user_test"
    assert "failure_reason" in d

    restored = DriverBaseline.from_dict(d)
    assert restored.driver_id == "user_test"
    assert restored.is_valid is False

    calibrator = BaselineCalibrator()
    calibrator.start_calibration()
    calibrator.add_sample(1.0, 0.3, True)
    assert calibrator.state == CalibrationState.COLLECTING
    calibrator.reset()
    assert calibrator.state == CalibrationState.IDLE
    assert len(calibrator._samples) == 0
