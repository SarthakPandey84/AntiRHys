"""
AntiRHys Phase 12.4 — Closure Continuity Audit Regression Test Suite
=====================================================================
Deterministic edge-case audit tests for tracking-gap closure accumulation,
grace-period boundaries, blink resets, recovery interruption, and timing arithmetic.

Covers the 10 Phase 12.4 audit requirements:
1. Continuous closure lasting just below, exactly at, and just above 1.5 seconds.
2. Short tracking gap followed by reopening rather than continued closure.
3. Multiple short gaps within one sustained closure.
4. Multiple short gaps separating distinct brief closure events.
5. Gap just below, exactly at, and just above the 300 ms grace period.
6. Long tracking loss followed by reacquisition with closed eyes.
7. Long tracking loss followed by reacquisition with open eyes.
8. Alarm recovery interrupted by tracking loss.
9. Detector reset or session stop while closure is pending.
10. Non-monotonic timestamps and unexpectedly large timestamp jumps.
"""

import pytest
from detector import (
    DrowsinessDetector,
    DetectionConfig,
    DetectionState,
    AlarmState,
    TrackingState,
    EyeState,
)

OPEN_EYE_LANDMARKS = [
    (10.0, 50.0),
    (16.0, 43.0),
    (24.0, 43.0),
    (30.0, 50.0),
    (24.0, 50.0),
    (16.0, 50.0),
]

CLOSED_EYE_LANDMARKS = [
    (10.0, 50.0),
    (16.0, 49.5),
    (24.0, 49.5),
    (30.0, 50.0),
    (24.0, 50.5),
    (16.0, 50.5),
]


def test_edge_case_1_continuous_closure_just_below_exact_just_above_1_5s():
    """
    Edge Case 1: Continuous closure lasting just below, exactly at, and just above 1.5 seconds.
    - Below (1.49s): AlarmState.OFF, DetectionState.NORMAL.
    - Exact (1.50s): AlarmState.TRIGGERED, DetectionState.POSSIBLE_DROWSINESS.
    - Above (1.51s): AlarmState.TRIGGERED, DetectionState.POSSIBLE_DROWSINESS.
    """
    cfg = DetectionConfig(closure_duration_threshold=1.5, ear_threshold=0.25)
    det = DrowsinessDetector("audit_ec1", config=cfg)

    # Start closure at t = 10.0
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=10.0)

    # 1a. Just below (1.49s closure duration)
    res_below = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=11.49)
    assert res_below.closure_duration == pytest.approx(1.49, abs=1e-4)
    assert res_below.alarm_state == AlarmState.OFF
    assert res_below.detection_state == DetectionState.NORMAL

    # 1b. Exactly at threshold (1.50s closure duration)
    res_exact = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=11.50)
    assert res_exact.closure_duration == pytest.approx(1.50, abs=1e-4)
    assert res_exact.alarm_state == AlarmState.TRIGGERED
    assert res_exact.detection_state == DetectionState.POSSIBLE_DROWSINESS

    # 1c. Just above threshold (1.51s closure duration)
    res_above = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=11.51)
    assert res_above.closure_duration == pytest.approx(1.51, abs=1e-4)
    assert res_above.alarm_state == AlarmState.TRIGGERED
    assert res_above.detection_state == DetectionState.POSSIBLE_DROWSINESS


def test_edge_case_2_short_tracking_gap_followed_by_reopening():
    """
    Edge Case 2: Short tracking gap followed by reopening rather than continued closure.
    - 2a: Normal blink interrupted by gap: 0.15s closed -> 0.10s gap -> open.
          Asserts blink_count increments, alarm stays OFF, closure resets.
    - 2b: Non-blink closure: 0.60s closed -> 0.10s gap -> open.
          Asserts blink_count NOT incremented (0.60s > 0.45s max), alarm stays OFF, closure resets.
    """
    cfg = DetectionConfig(closure_duration_threshold=1.5, tracking_gap_grace_sec=0.30)

    # 2a: Normal blink with gap
    det_a = DrowsinessDetector("audit_ec2a", config=cfg)
    det_a.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=20.0)
    det_a.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=20.15)
    det_a.process_frame_landmarks(False, now=20.25)
    res_reopen_a = det_a.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=20.35)

    assert res_reopen_a.blink_count == 1
    assert res_reopen_a.alarm_state == AlarmState.OFF
    assert res_reopen_a.closure_duration == 0.0
    assert det_a.closure_duration == 0.0

    # 2b: Prolonged non-blink closure with gap followed by reopen
    det_b = DrowsinessDetector("audit_ec2b", config=cfg)
    det_b.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=30.0)
    det_b.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=30.60)
    det_b.process_frame_landmarks(False, now=30.70)
    res_reopen_b = det_b.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=30.80)

    assert res_reopen_b.blink_count == 0  # 0.60s > 0.45s max blink
    assert res_reopen_b.alarm_state == AlarmState.OFF
    assert res_reopen_b.closure_duration == 0.0
    assert det_b.closure_duration == 0.0


def test_edge_case_3_multiple_short_gaps_within_one_sustained_closure():
    """
    Edge Case 3: Multiple short gaps within one sustained closure.
    Sequence:
    - Seg 1: 0.5s closed (t=40.0 to 40.5)
    - Gap 1: 0.1s gap (t=40.6)
    - Seg 2: 0.5s closed (t=40.7 to 41.2, observed=1.0s)
    - Gap 2: 0.1s gap (t=41.3)
    - Seg 3: 0.4s closed (t=41.4 to 41.8, observed=1.4s) -> Alarm OFF
    - Seg 3+: 0.5s closed (t=41.9, observed=1.5s) -> Alarm TRIGGERS
    - Gap 3: 0.1s gap (t=42.0) -> Alarm remains TRIGGERED
    - Post-alarm frame (t=42.1) -> Alarm remains TRIGGERED
    """
    cfg = DetectionConfig(closure_duration_threshold=1.5, tracking_gap_grace_sec=0.30)
    det = DrowsinessDetector("audit_ec3", config=cfg)

    # Seg 1: 0.5s closed
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=40.0)
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=40.5)
    # Gap 1: 0.1s
    det.process_frame_landmarks(False, now=40.6)

    # Seg 2: 0.5s closed (cumulative observed = 1.0s)
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=40.7)
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=41.2)
    # Gap 2: 0.1s
    det.process_frame_landmarks(False, now=41.3)

    # Seg 3: 0.4s closed (cumulative observed = 1.4s < 1.5s)
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=41.4)
    res_1_4 = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=41.8)
    assert res_1_4.closure_duration == pytest.approx(1.40, abs=1e-4)
    assert res_1_4.alarm_state == AlarmState.OFF

    # Seg 3: reaches 1.5s observed closure at t = 41.9
    res_alarm = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=41.9)
    assert res_alarm.closure_duration == pytest.approx(1.50, abs=1e-4)
    assert res_alarm.alarm_state == AlarmState.TRIGGERED
    assert res_alarm.detection_state == DetectionState.POSSIBLE_DROWSINESS

    # Gap 3 while alarm is active
    res_gap3 = det.process_frame_landmarks(False, now=42.0)
    assert res_gap3.alarm_state == AlarmState.TRIGGERED
    assert res_gap3.detection_state == DetectionState.TRACKING_LOST

    # Resumed after gap 3: first frame at t=42.1 confirms resumed closure at 1.50s
    res_resume = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=42.1)
    assert res_resume.alarm_state == AlarmState.TRIGGERED
    assert res_resume.closure_duration == pytest.approx(1.50, abs=1e-4)

    # Next frame at t=42.2 advances closure duration to 1.60s
    res_advance = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=42.2)
    assert res_advance.alarm_state == AlarmState.TRIGGERED
    assert res_advance.closure_duration == pytest.approx(1.60, abs=1e-4)


def test_edge_case_4_multiple_short_gaps_separating_distinct_brief_closures():
    """
    Edge Case 4: Multiple short gaps separating distinct brief closure events.
    Verifies that brief closures separated by open-eye driving NEVER accumulate into an acute alarm.
    - Event 1: Closed 0.2s -> gap 0.1s -> Reopen at t=50.4 (blink 1, accumulator cleared).
    - Attentive Driving: Eyes open for 1.6s (t=50.4 to 52.0).
    - Event 2: Closed 0.2s -> gap 0.1s -> Reopen at t=52.4 (blink 2, accumulator cleared).
    Asserts: Total blinks = 2, AlarmState.OFF throughout, closure_duration = 0.0.
    """
    cfg = DetectionConfig(closure_duration_threshold=1.5, tracking_gap_grace_sec=0.30)
    det = DrowsinessDetector("audit_ec4", config=cfg)

    # Event 1
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=50.0)
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=50.2)
    det.process_frame_landmarks(False, now=50.3)
    res_ev1 = det.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=50.4)
    assert res_ev1.blink_count == 1
    assert res_ev1.alarm_state == AlarmState.OFF
    assert det.closure_duration == 0.0

    # Attentive driving (eyes continuously open)
    det.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=51.0)
    det.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=52.0)
    assert det.closure_duration == 0.0

    # Event 2
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=52.0)
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=52.2)
    det.process_frame_landmarks(False, now=52.3)
    res_ev2 = det.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=52.4)
    assert res_ev2.blink_count == 2
    assert res_ev2.alarm_state == AlarmState.OFF
    assert det.closure_duration == 0.0


def test_edge_case_5_grace_period_boundaries_below_exact_above_300ms():
    """
    Edge Case 5: A gap just below (290ms), exactly at (300ms), and just above (310ms) the 300ms grace period.
    - 5a (290ms gap < 300ms): Closure continuity PRESERVED -> Alarm triggers when total observed closure = 1.5s.
    - 5b (300ms gap == 300ms): Grace period boundary ACCEPTED -> Alarm triggers when total observed closure = 1.5s.
    - 5c (310ms gap > 300ms): Grace period EXCEEDED -> Prior closure discarded -> Alarm stays OFF.
    """
    cfg = DetectionConfig(closure_duration_threshold=1.5, tracking_gap_grace_sec=0.30)

    # 5a: Gap = 0.29s (290ms)
    det_a = DrowsinessDetector("audit_ec5a", config=cfg)
    det_a.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=60.0)
    det_a.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=61.0)
    det_a.process_frame_landmarks(False, now=61.15)
    # Re-acquire at 61.29 (gap = 61.29 - 61.0 = 0.29s <= 0.30s)
    det_a.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=61.29)
    res_a = det_a.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=61.79)
    assert res_a.closure_duration == pytest.approx(1.50, abs=1e-4)
    assert res_a.alarm_state == AlarmState.TRIGGERED

    # 5b: Gap = 0.30s (300ms exact boundary)
    det_b = DrowsinessDetector("audit_ec5b", config=cfg)
    det_b.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=70.0)
    det_b.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=71.0)
    det_b.process_frame_landmarks(False, now=71.15)
    # Re-acquire at 71.30 (gap = 71.30 - 71.0 = 0.30s == 0.30s)
    det_b.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=71.30)
    res_b = det_b.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=71.80)
    assert res_b.closure_duration == pytest.approx(1.50, abs=1e-4)
    assert res_b.alarm_state == AlarmState.TRIGGERED

    # 5c: Gap = 0.31s (310ms, exceeds grace period)
    det_c = DrowsinessDetector("audit_ec5c", config=cfg)
    det_c.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=80.0)
    det_c.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=81.0)
    det_c.process_frame_landmarks(False, now=81.15)
    # Re-acquire at 81.31 (gap = 81.31 - 81.0 = 0.31s > 0.30s)
    det_c.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=81.31)
    res_c = det_c.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=81.91)
    # Prior 1.0s was discarded; only post-gap 0.60s accumulated
    assert res_c.closure_duration == pytest.approx(0.60, abs=1e-4)
    assert res_c.alarm_state == AlarmState.OFF
    assert res_c.detection_state == DetectionState.NORMAL


def test_edge_case_6_long_tracking_loss_followed_by_closed_eyes_reacquisition():
    """
    Edge Case 6: Long tracking loss followed by reacquisition with closed eyes.
    - Pre-loss closure: 1.0s (t=90.0 to 91.0).
    - Long tracking loss: 3.0s (t=91.1 to 94.0).
    - Reacquisition with closed eyes for 0.6s (t=94.1 to 94.7).
    Asserts: Prior closure is NOT reused; closure_duration = 0.60s; AlarmState.OFF.
    """
    cfg = DetectionConfig(closure_duration_threshold=1.5, tracking_gap_grace_sec=0.30)
    det = DrowsinessDetector("audit_ec6", config=cfg)

    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=90.0)
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=91.0)

    # 3.0s loss
    for i in range(1, 31):
        det.process_frame_landmarks(False, now=91.0 + i * 0.1)

    # Re-acquire closed
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=94.1)
    res = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=94.7)

    assert res.closure_duration == pytest.approx(0.60, abs=1e-4)
    assert res.alarm_state == AlarmState.OFF
    assert res.detection_state == DetectionState.NORMAL


def test_edge_case_7_long_tracking_loss_followed_by_open_eyes_reacquisition():
    """
    Edge Case 7: Long tracking loss followed by reacquisition with open eyes.
    - Pre-loss closure: 1.0s (t=100.0 to 101.0).
    - Long tracking loss: 3.0s (t=101.1 to 104.0).
    - Reacquisition with open eyes at t=104.1.
    Asserts: Stale closure discarded; blink count = 0; closure_duration = 0.0; state NORMAL.
    """
    cfg = DetectionConfig(closure_duration_threshold=1.5, tracking_gap_grace_sec=0.30)
    det = DrowsinessDetector("audit_ec7", config=cfg)

    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=100.0)
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=101.0)

    for i in range(1, 31):
        det.process_frame_landmarks(False, now=101.0 + i * 0.1)

    res = det.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=104.1)
    assert res.blink_count == 0
    assert res.alarm_state == AlarmState.OFF
    assert res.closure_duration == 0.0
    assert res.tracking_state == TrackingState.ACTIVE
    assert res.detection_state == DetectionState.NORMAL


def test_edge_case_8_alarm_recovery_interrupted_by_tracking_loss():
    """
    Edge Case 8: Alarm recovery interrupted by tracking loss.
    Sequence:
    1. 1.6s closure triggers alarm (t=110.0 to 111.6).
    2. Eyes reopen for 0.5s (t=111.7 to 112.2, open duration = 0.5s < 1.0s recovery).
       Alarm MUST stay TRIGGERED.
    3. Tracking is lost for 0.4s (t=112.3 to 112.7).
       Alarm MUST stay TRIGGERED in TRACKING_LOST.
    4. Face returns with OPEN eyes at t=112.8.
       Driver must maintain continuous open eyes for full 1.0s recovery.
       - t=113.3 (0.5s open): Still TRIGGERED (verifying alertness).
       - t=113.8 (1.0s open): Clears to AlarmState.OFF and DetectionState.NORMAL.
    """
    cfg = DetectionConfig(closure_duration_threshold=1.5, recovery_open_duration=1.0)
    det = DrowsinessDetector("audit_ec8", config=cfg)

    # 1. Trigger alarm
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=110.0)
    res_trig = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=111.6)
    assert res_trig.alarm_state == AlarmState.TRIGGERED

    # 2. Reopen for 0.5s (partial recovery)
    det.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=111.7)
    res_part = det.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=112.2)
    assert res_part.alarm_state == AlarmState.TRIGGERED
    assert res_part.detection_state == DetectionState.POSSIBLE_DROWSINESS

    # 3. Tracking lost during recovery
    res_lost = det.process_frame_landmarks(False, now=112.4)
    assert res_lost.alarm_state == AlarmState.TRIGGERED
    assert res_lost.detection_state == DetectionState.TRACKING_LOST

    # 4. Face returns open; recovery timer restarts
    det.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=112.8)
    res_mid = det.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=113.3)
    assert res_mid.alarm_state == AlarmState.TRIGGERED

    res_recovered = det.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=113.8)
    assert res_recovered.alarm_state == AlarmState.OFF
    assert res_recovered.detection_state == DetectionState.NORMAL


def test_edge_case_9_session_reset_and_stop_during_pending_closure():
    """
    Edge Case 9: Detector reset or session stop while closure is pending.
    Verifies that calling reset_session() or stop_session() cleanly wipes all accumulated
    closure state and prevents stale closure leak into subsequent frames.
    """
    det = DrowsinessDetector("audit_ec9")

    # 1. Accumulate 1.2s closure
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=120.0)
    res_pend = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=121.2)
    assert res_pend.closure_duration == pytest.approx(1.20, abs=1e-4)

    # 2. Reset session while closure pending
    det.reset_session("audit_ec9_fresh")
    assert det.closure_duration == 0.0
    assert det.alarm_state == AlarmState.OFF
    assert det.blink_count == 0

    # Next frame arrives closed in new session -> starts from 0.0, not 1.3s
    res_fresh = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=121.3)
    assert res_fresh.closure_duration == 0.0
    assert res_fresh.alarm_state == AlarmState.OFF

    # 3. Accumulate in fresh session then stop_session()
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=122.0)
    det.stop_session()
    assert det.is_active is False
    assert det.closure_duration == 0.0
    assert det.alarm_state == AlarmState.OFF

    # Processing while stopped yields UNAVAILABLE
    res_stopped = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=122.5)
    assert res_stopped.detection_state == DetectionState.UNAVAILABLE
    assert res_stopped.alarm_state == AlarmState.OFF


def test_edge_case_10_non_monotonic_and_large_timestamp_jumps():
    """
    Edge Case 10: Non-monotonic timestamps and unexpectedly large timestamp jumps.
    - 10a: Non-monotonic timestamp (now < last_processed_time) is safely clamped to last_processed_time.
    - 10b: Non-monotonic timestamp in process_frame_error() is clamped to last_processed_time.
    - 10c: Large jump forward (10s) across tracking loss resets closure accumulation.
    """
    det = DrowsinessDetector("audit_ec10")

    # 10a: Backward jump in process_frame_landmarks
    det.process_frame_landmarks(True, OPEN_EYE_LANDMARKS, OPEN_EYE_LANDMARKS, now=140.0)
    det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=140.5)
    # Time jumps backward to 139.0
    res_back = det.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=139.0)
    assert res_back.timestamp == 140.5  # Clamped to last processed time
    assert res_back.closure_duration >= 0.0  # Does not go negative

    # 10b: Backward jump in process_frame_error
    res_err = det.process_frame_error("frame_corrupt", now=138.0)
    assert res_err.timestamp == 140.5  # Clamped

    # 10c: Large jump forward across tracking loss (10s)
    det_jump = DrowsinessDetector("audit_ec10c")
    det_jump.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=150.0)
    det_jump.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=150.5)
    # Tracking loss starts at 150.6
    det_jump.process_frame_landmarks(False, now=150.6)
    # Jump 10s forward with tracking lost (160.5 - 150.6 = 9.9s > 0.8s timeout)
    res_jump_lost = det_jump.process_frame_landmarks(False, now=160.5)
    assert res_jump_lost.tracking_state == TrackingState.LOST

    # Return closed at 160.6 -> Starts from 0.0, prior 0.5s discarded
    res_jump_return = det_jump.process_frame_landmarks(True, CLOSED_EYE_LANDMARKS, CLOSED_EYE_LANDMARKS, now=160.6)
    assert res_jump_return.closure_duration == 0.0
    assert res_jump_return.alarm_state == AlarmState.OFF
