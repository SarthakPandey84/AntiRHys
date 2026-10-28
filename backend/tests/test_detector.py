"""
Unit tests for AntiRHys Drowsiness Detector and EAR calculations.
"""

import pytest
from detector import (
    calculate_ear,
    DrowsinessDetector,
    DetectionConfig,
    DetectionState,
    AlarmState,
    TrackingState,
    EyeState
)


# Helper fixtures: synthetic eye coordinates
# Open eye landmarks: horizontal width = 20, vertical openings = 7 and 7 -> EAR = (7+7)/(2*20) = 0.35
OPEN_EYE_LANDMARKS = [
    (10.0, 50.0),  # p1: lateral corner
    (16.0, 43.0),  # p2: upper 1
    (24.0, 43.0),  # p3: upper 2
    (30.0, 50.0),  # p4: medial corner
    (24.0, 50.0),  # p5: lower 2
    (16.0, 50.0),  # p6: lower 1
]

# Closed eye landmarks: horizontal width = 20, vertical openings = 1 and 1 -> EAR = (1+1)/(2*20) = 0.05
CLOSED_EYE_LANDMARKS = [
    (10.0, 50.0),  # p1: lateral corner
    (16.0, 49.5),  # p2: upper 1
    (24.0, 49.5),  # p3: upper 2
    (30.0, 50.0),  # p4: medial corner
    (24.0, 50.5),  # p5: lower 2
    (16.0, 50.5),  # p6: lower 1
]


def test_calculate_ear_open_eye():
    ear = calculate_ear(OPEN_EYE_LANDMARKS)
    assert 0.30 <= ear <= 0.40


def test_calculate_ear_closed_eye():
    ear = calculate_ear(CLOSED_EYE_LANDMARKS)
    assert 0.01 <= ear <= 0.15


def test_calculate_ear_degenerate_zero_width():
    # p1 and p4 identical
    degenerate_landmarks = [(50.0, 50.0)] * 6
    assert calculate_ear(degenerate_landmarks, min_width=1.0) == 0.0


def test_calculate_ear_invalid_length():
    # Only 5 points
    assert calculate_ear([(10.0, 10.0)] * 5) == 0.0


def test_calculate_ear_non_finite_values():
    bad_landmarks = OPEN_EYE_LANDMARKS.copy()
    bad_landmarks[0] = (float('inf'), 50.0)
    assert calculate_ear(bad_landmarks) == 0.0


def test_detector_normal_open_eyes():
    detector = DrowsinessDetector(session_id="test_sess")
    t0 = 100.0

    res = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=OPEN_EYE_LANDMARKS,
        right_eye_points=OPEN_EYE_LANDMARKS,
        now=t0
    )

    assert res.detection_state == DetectionState.NORMAL
    assert res.alarm_state == AlarmState.OFF
    assert res.tracking_state == TrackingState.ACTIVE
    assert res.eye_state == EyeState.OPEN
    assert res.ear >= 0.25


def test_detector_prolonged_eye_closure_triggers_alarm():
    config = DetectionConfig(
        ear_threshold=0.25,
        closure_duration_threshold=1.5,
        recovery_open_duration=1.0
    )
    detector = DrowsinessDetector(session_id="test_sess", config=config)
    t = 100.0

    # Frame 1: Eyes initially open
    res1 = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=OPEN_EYE_LANDMARKS,
        right_eye_points=OPEN_EYE_LANDMARKS,
        now=t
    )
    assert res1.detection_state == DetectionState.NORMAL

    # Frame 2: Eyes close at t = 100.1
    t += 0.1
    res2 = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=CLOSED_EYE_LANDMARKS,
        right_eye_points=CLOSED_EYE_LANDMARKS,
        now=t
    )
    assert res2.detection_state == DetectionState.NORMAL  # Has not met 1.5s threshold yet
    assert res2.alarm_state == AlarmState.OFF
    assert res2.eye_state == EyeState.CLOSED

    # Frame 3: Eyes still closed at t = 101.0 (0.9s closed)
    t = 101.0
    res3 = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=CLOSED_EYE_LANDMARKS,
        right_eye_points=CLOSED_EYE_LANDMARKS,
        now=t
    )
    assert res3.detection_state == DetectionState.NORMAL
    assert res3.alarm_state == AlarmState.OFF

    # Frame 4: Eyes still closed at t = 101.6 (1.5s closed) -> Drowsiness triggers!
    t = 101.6
    res4 = detector.process_frame_landmarks(
        has_face=True,
        left_eye_points=CLOSED_EYE_LANDMARKS,
        right_eye_points=CLOSED_EYE_LANDMARKS,
        now=t
    )
    assert res4.detection_state == DetectionState.POSSIBLE_DROWSINESS
    assert res4.alarm_state == AlarmState.TRIGGERED


def test_detector_recovery_hysteresis():
    config = DetectionConfig(
        ear_threshold=0.25,
        closure_duration_threshold=1.5,
        recovery_open_duration=1.0
    )
    detector = DrowsinessDetector(session_id="test_sess", config=config)
    t = 100.0

    # Start closure
    detector.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=t)
    # Trigger drowsiness
    t += 1.6
    res = detector.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=t)
    assert res.alarm_state == AlarmState.TRIGGERED

    # Eyes reopen at t + 0.1
    t += 0.1
    res_reopen = detector.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=t)
    # Immediate 1-frame reopen must NOT instantly clear the alarm (hysteresis active)
    assert res_reopen.alarm_state == AlarmState.TRIGGERED

    # Eyes remain open for 0.5s (still under 1.0s recovery duration)
    t += 0.5
    res_half_recovery = detector.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=t)
    assert res_half_recovery.alarm_state == AlarmState.TRIGGERED

    # Eyes remain open past 1.0s (t + 1.1s) -> Alarm clears cleanly
    t += 0.6
    res_recovered = detector.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=t)
    assert res_recovered.alarm_state == AlarmState.OFF
    assert res_recovered.detection_state == DetectionState.NORMAL


def test_detector_normal_blink_counting():
    config = DetectionConfig(
        blink_min_duration=0.08,
        blink_max_duration=0.45,
        closure_duration_threshold=1.5
    )
    detector = DrowsinessDetector(session_id="test_sess", config=config)
    t = 10.0

    # Frame 1: Open
    detector.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=t)
    assert detector.blink_count == 0

    # Frame 2: Closed for 0.20s (normal blink)
    t += 0.1
    detector.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=t)
    assert detector.blink_count == 0

    # Frame 3: Eyes reopen at t + 0.20s
    t += 0.20
    res = detector.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=t)
    assert res.blink_count == 1
    assert res.alarm_state == AlarmState.OFF


def test_detector_missing_face_does_not_claim_alert():
    config = DetectionConfig(tracking_lost_timeout=0.8)
    detector = DrowsinessDetector(session_id="test_sess", config=config)
    t = 50.0

    # Initial frame with face
    detector.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=t)

    # Face disappears
    t += 0.1
    res1 = detector.process_frame_landmarks(False, now=t)
    assert res1.tracking_state == TrackingState.SEARCHING
    assert res1.eye_state == EyeState.UNKNOWN

    # Face continues missing beyond 0.8s
    t += 0.9
    res2 = detector.process_frame_landmarks(False, now=t)
    assert res2.tracking_state == TrackingState.LOST
    assert res2.detection_state == DetectionState.TRACKING_LOST
    # MUST NOT be NORMAL / Alert
    assert res2.detection_state != DetectionState.NORMAL


def test_detector_session_reset_and_stop():
    detector = DrowsinessDetector(session_id="sess_1")
    t = 10.0
    detector.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=t)
    t += 2.0
    res = detector.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=t)
    assert res.alarm_state == AlarmState.TRIGGERED

    # Reset
    detector.reset_session("sess_2")
    assert detector.session_id == "sess_2"
    assert detector.alarm_state == AlarmState.OFF
    assert detector.blink_count == 0

    # Stop session
    detector.stop_session()
    stopped_res = detector.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=t)
    assert stopped_res.detection_state == DetectionState.UNAVAILABLE
    assert stopped_res.alarm_state == AlarmState.OFF


def test_invalid_geometry_between_closed_frames_resets_closure_timer():
    """
    Ensure invalid eye landmark geometry cannot cause unknown time to count
    towards continuous eye closure.
    """
    config = DetectionConfig(closure_duration_threshold=1.5)
    detector = DrowsinessDetector(session_id="test_sess", config=config)
    t = 100.0

    # 1. Closed eyes for 0.5s (t = 100.0 to 100.5)
    detector.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=t)
    t = 100.5
    res1 = detector.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=t)
    assert res1.closure_duration == pytest.approx(0.5, 0.05)
    assert res1.alarm_state == AlarmState.OFF

    # 2. Invalid/degenerate landmark geometry arrives at t = 100.6 (both eyes 0 width)
    t = 100.6
    degenerate = [(50.0, 50.0)] * 6
    res_invalid = detector.process_frame_landmarks(True, degenerate, degenerate, now=t)
    assert res_invalid.eye_state == EyeState.UNKNOWN
    assert res_invalid.closure_duration == 0.0

    # 3. Eyes closed again at t = 101.6 (1.6s since initial t0, but only 1.0s since invalid frame)
    t = 101.6
    res_reclosed = detector.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=t)
    # MUST NOT trigger drowsiness because the invalid frame reset the closure timer
    assert res_reclosed.alarm_state == AlarmState.OFF
    assert res_reclosed.detection_state == DetectionState.NORMAL
    assert res_reclosed.closure_duration == pytest.approx(0.0, 0.05)


def test_tracking_loss_while_drowsiness_alarm_active():
    """
    Verify defined alarm behavior when tracking is unavailable:
    - If drowsiness alarm was triggered, tracking loss reports TRACKING_LOST while preserving TRIGGERED alarm.
    - Alarm remains sounding until face is re-acquired and eyes remain open for recovery duration.
    """
    config = DetectionConfig(closure_duration_threshold=1.5, recovery_open_duration=1.0, tracking_lost_timeout=0.8)
    detector = DrowsinessDetector(session_id="test_sess", config=config)
    t = 100.0

    # 1. Trigger drowsiness alert (1.6s closed)
    detector.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=t)
    t = 101.6
    res_drowsy = detector.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=t)
    assert res_drowsy.alarm_state == AlarmState.TRIGGERED
    assert res_drowsy.detection_state == DetectionState.POSSIBLE_DROWSINESS

    # 2. Face tracking is lost at t = 101.7
    t = 101.7
    res_lost1 = detector.process_frame_landmarks(False, now=t)
    assert res_lost1.tracking_state == TrackingState.SEARCHING
    # Alarm must persist because driver was confirmed drowsy before losing tracking
    assert res_lost1.alarm_state == AlarmState.TRIGGERED
    assert res_lost1.detection_state == DetectionState.TRACKING_LOST

    # 3. Face tracking remains lost past timeout at t = 102.6 (0.9s lost)
    t = 102.6
    res_lost2 = detector.process_frame_landmarks(False, now=t)
    assert res_lost2.tracking_state == TrackingState.LOST
    assert res_lost2.detection_state == DetectionState.TRACKING_LOST
    assert res_lost2.alarm_state == AlarmState.TRIGGERED

    # 4. Face re-appears with eyes open at t = 103.0
    t = 103.0
    res_returned = detector.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=t)
    assert res_returned.tracking_state == TrackingState.ACTIVE
    # Alarm must remain triggered and state must be POSSIBLE_DROWSINESS until recovery open duration (1.0s) is satisfied
    assert res_returned.alarm_state == AlarmState.TRIGGERED
    assert res_returned.detection_state == DetectionState.POSSIBLE_DROWSINESS

    # 5. Eyes open for only 0.5s at t = 103.5 -> Still triggered and POSSIBLE_DROWSINESS
    t = 103.5
    res_half = detector.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=t)
    assert res_half.alarm_state == AlarmState.TRIGGERED
    assert res_half.detection_state == DetectionState.POSSIBLE_DROWSINESS

    # 6. Eyes open for 1.1s at t = 104.1 -> Recovery satisfied, alarm clears
    t = 104.1
    res_cleared = detector.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=t)
    assert res_cleared.alarm_state == AlarmState.OFF
    assert res_cleared.detection_state == DetectionState.NORMAL
