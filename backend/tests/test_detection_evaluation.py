"""
AntiRHys Phase 12.2 — Detection Evaluation Regression Test Suite
================================================================
Deterministic synthetic evaluation tests covering:
- Scenario A: Normal attentive behavior & blink counting
- Scenario B: Prolonged eye closure trigger latency (1.0s, 1.5s, 1.6s, 2.0s, 3.0s)
- Scenario C: Recovery hysteresis (nominal clearance & interrupted recovery)
- Scenario D: Boundary EAR noise oscillation & blink inflation
- Scenario E: Tracking loss resets closure timer
- Scenario F: PERCLOS observation gaps & time integration
- Scenario G: Calibration quality gates & clamping bounds
"""

import pytest
from detector import DetectionConfig, DetectionState, AlarmState, TrackingState
from evaluate_detection import (
    ScenarioStep,
    run_scenario,
    build_normal_attentive_steps,
    build_prolonged_closure_steps,
    build_recovery_hysteresis_steps,
    build_boundary_noise_steps,
    build_tracking_loss_during_closure_steps,
    build_perclos_gap_steps,
    evaluate_calibration_scenario,
)
from baseline_calibrator import CalibrationFailureReason


def test_scenario_a_normal_attentive_behavior():
    """
    Scenario A: Attentive driving with periodic natural blinks.
    Asserts:
    1. Zero false drowsiness alarms across 60 seconds of driving.
    2. Exact blink count matches injected blinks (15 blinks @ 1 blink / 4.0s).
    3. Final PERCLOS status is VALID with cumulative closure ~5% (well below 20% warning threshold).
    """
    steps = build_normal_attentive_steps(duration_sec=60.0, fps=10.0, blink_interval=4.0, blink_duration=0.20)
    result = run_scenario("scenario_a_attentive", steps)

    assert result.alarm_triggered is False, (
        f"Expected 0 drowsiness alarms during normal attentive driving, but got {result.total_alarm_frames} alarm frames."
    )
    assert result.total_blinks == 15, (
        f"Expected exactly 15 blinks detected for 60s sequence with 4s intervals, got {result.total_blinks}."
    )
    assert result.final_perclos_valid is True, (
        "Expected PERCLOS to be VALID after 60s of continuous valid tracking."
    )
    assert result.final_perclos == pytest.approx(3.0 / 60.0, abs=0.015), (
        f"Expected PERCLOS ~5.0% for 15 x 0.2s blinks, got {result.final_perclos * 100:.2f}%."
    )


def test_scenario_b_prolonged_eye_closure_trigger_latency():
    """
    Scenario B: Prolonged closures of 1.0s, 1.5s, 1.6s, 2.0s, and 3.0s.
    Asserts:
    1. 1.0s closure does NOT trigger drowsiness alarm (below 1.5s threshold).
    2. 1.5s closure (15 frames @ 10 FPS spanning t=0.0 to t=1.4) does not trigger because
       at t=1.5s the eye reopens; 16 closed frames spanning t=0.0 to t=1.5s (1.6s closure)
       triggers on exactly the 16th closed frame (trigger delay = 1.50s).
    3. Closures of 2.0s and 3.0s trigger with exactly 1.50s delay from closure onset.
    """
    # 1.0s closure: should not trigger
    steps_1_0, onset_1_0 = build_prolonged_closure_steps(closure_duration=1.0, fps=10.0)
    res_1_0 = run_scenario("b_1.0s", steps_1_0, stimulus_onset_time=onset_1_0)
    assert res_1_0.alarm_triggered is False, "1.0s closure unexpectedly triggered an alarm."

    # 1.5s closure (15 frames: elapsed closed time reaches 1.4s before reopening)
    steps_1_5, onset_1_5 = build_prolonged_closure_steps(closure_duration=1.5, fps=10.0)
    res_1_5 = run_scenario("b_1.5s", steps_1_5, stimulus_onset_time=onset_1_5)
    assert res_1_5.alarm_triggered is False, (
        "1.5s closure with 15 frames @ 10 FPS reached elapsed 1.4s and should not trigger."
    )

    # 1.6s closure (16 frames: 16th frame is at t = onset + 1.5s >= 1.5s threshold)
    steps_1_6, onset_1_6 = build_prolonged_closure_steps(closure_duration=1.6, fps=10.0)
    res_1_6 = run_scenario("b_1.6s", steps_1_6, stimulus_onset_time=onset_1_6)
    assert res_1_6.alarm_triggered is True, "1.6s closure failed to trigger drowsiness alarm."
    assert res_1_6.trigger_delay_sec == pytest.approx(1.5, abs=0.01), (
        f"Expected trigger delay of exactly 1.5s, got {res_1_6.trigger_delay_sec}s."
    )

    # 2.0s closure
    steps_2_0, onset_2_0 = build_prolonged_closure_steps(closure_duration=2.0, fps=10.0)
    res_2_0 = run_scenario("b_2.0s", steps_2_0, stimulus_onset_time=onset_2_0)
    assert res_2_0.alarm_triggered is True
    assert res_2_0.trigger_delay_sec == pytest.approx(1.5, abs=0.01)

    # 3.0s closure
    steps_3_0, onset_3_0 = build_prolonged_closure_steps(closure_duration=3.0, fps=10.0)
    res_3_0 = run_scenario("b_3.0s", steps_3_0, stimulus_onset_time=onset_3_0)
    assert res_3_0.alarm_triggered is True
    assert res_3_0.trigger_delay_sec == pytest.approx(1.5, abs=0.01)


def test_scenario_c_recovery_hysteresis_and_interrupted_recovery():
    """
    Scenario C: Alarm clearance requires 1.0s of continuous open eyes.
    Asserts:
    1. Nominal: After alarm triggers, eyes reopen; alarm clears after exactly 1.0s (10 open frames).
    2. Interrupted: If eyes reopen for 0.5s (< 1.0s) and close again, the alarm does NOT clear.
    """
    # Nominal recovery
    steps_nom, onset_nom, reopen_nom = build_recovery_hysteresis_steps(
        closure_duration=2.0,
        open_recovery_duration=2.5,
        fps=10.0
    )
    res_nom = run_scenario("c_nominal", steps_nom, stimulus_onset_time=onset_nom, recovery_onset_time=reopen_nom)
    assert res_nom.alarm_triggered is True
    assert res_nom.recovery_cleared_timestamp is not None, "Alarm failed to clear during recovery."
    assert res_nom.recovery_delay_sec == pytest.approx(1.0, abs=0.01), (
        f"Expected recovery delay of 1.0s, got {res_nom.recovery_delay_sec}s."
    )

    # Interrupted recovery (eyes reopen for only 0.5s then close again)
    steps_int, onset_int, _ = build_recovery_hysteresis_steps(
        closure_duration=2.0,
        interrupt_after_sec=0.5,
        fps=10.0
    )
    res_int = run_scenario("c_interrupted", steps_int, stimulus_onset_time=onset_int)
    assert res_int.alarm_triggered is True
    assert res_int.recovery_cleared_timestamp is None, (
        "Alarm prematurely cleared despite recovery being interrupted after 0.5s."
    )


def test_scenario_d_boundary_noise_oscillation():
    """
    Scenario D: Boundary EAR noise oscillating between 0.245 and 0.255 at 10 Hz.
    Asserts:
    1. Zero acute drowsiness alarms (closures never reach 1.5s continuous duration).
    2. Rapid state switching across 0.250 threshold inflates the blink counter (25 blinks in 5s).
    """
    steps = build_boundary_noise_steps(duration_sec=5.0, fps=10.0, low_ear=0.245, high_ear=0.255)
    res = run_scenario("d_boundary_noise", steps)

    assert res.alarm_triggered is False, "Boundary oscillation falsely triggered a drowsiness alarm."
    assert res.total_blinks == 25, (
        f"Expected 25 artificial blinks from 50 alternating frames, got {res.total_blinks}."
    )


def test_scenario_e_tracking_gap_continuity_short_gaps_trigger_alarm():
    """
    Scenario E1: Brief tracking gaps (<= 300ms) preserve closed-eye continuity.
    Asserts:
    1. 1.0s closed + 0.1s gap + 0.6s closed -> triggers alarm.
    2. 1.0s closed + 0.2s gap + 0.6s closed -> triggers alarm.
    3. Unobserved gap time is NOT credited towards closure duration:
       Accumulated observed closure is exactly 1.0s + 0.6s = 1.6s (> 1.5s threshold).
       Trigger delay from closure onset includes the gap wall-clock time:
       - 0.1s gap: trigger delay = 1.7s (1.0s pre + 0.1s gap + 0.5s post to reach 1.5s + 0.1s onset discretization).
       - 0.2s gap: trigger delay = 1.8s (1.0s pre + 0.2s gap + 0.5s post to reach 1.5s + 0.1s onset discretization).
    """
    # 0.1s gap (within 300ms grace period)
    steps_0_1, onset_0_1 = build_tracking_loss_during_closure_steps(
        pre_loss_closure_sec=1.0,
        loss_duration_sec=0.1,
        post_loss_closure_sec=0.6,
        fps=10.0
    )
    res_0_1 = run_scenario("e_loss_0.1s", steps_0_1, stimulus_onset_time=onset_0_1)
    assert res_0_1.alarm_triggered is True, (
        "0.1s tracking gap should preserve closure continuity and trigger alarm."
    )
    assert res_0_1.trigger_delay_sec == pytest.approx(1.7, abs=0.02)

    # 0.2s gap (within 300ms grace period)
    steps_0_2, onset_0_2 = build_tracking_loss_during_closure_steps(
        pre_loss_closure_sec=1.0,
        loss_duration_sec=0.2,
        post_loss_closure_sec=0.6,
        fps=10.0
    )
    res_0_2 = run_scenario("e_loss_0.2s", steps_0_2, stimulus_onset_time=onset_0_2)
    assert res_0_2.alarm_triggered is True, (
        "0.2s tracking gap should preserve closure continuity and trigger alarm."
    )
    assert res_0_2.trigger_delay_sec == pytest.approx(1.8, abs=0.02)


def test_scenario_e_tracking_gap_exceeding_grace_period_resets_closure_continuity():
    """
    Scenario E2: Tracking gaps exceeding grace period (> 300ms) reset closure sequence.
    Asserts:
    1. 1.0s closed + 0.4s gap + 0.6s closed -> does NOT trigger alarm.
    2. Because gap (0.4s) > grace period (0.3s), the prior 1.0s closed time is discarded.
    3. Post-gap closure of 0.6s independently does not satisfy >= 1.5s threshold.
    """
    steps_0_4, onset_0_4 = build_tracking_loss_during_closure_steps(
        pre_loss_closure_sec=1.0,
        loss_duration_sec=0.4,
        post_loss_closure_sec=0.6,
        fps=10.0
    )
    res_0_4 = run_scenario("e_loss_0.4s", steps_0_4, stimulus_onset_time=onset_0_4)
    assert res_0_4.alarm_triggered is False, (
        "0.4s tracking gap exceeded 300ms grace period; closure continuity should reset without alarm."
    )


def test_scenario_e_repeated_short_gaps_preserve_continuity_and_trigger_alarm():
    """
    Scenario E3: Multiple consecutive short tracking gaps (each <= 300ms).
    Sequence: 0.6s closed -> 0.1s gap -> 0.5s closed -> 0.1s gap -> 0.5s closed.
    Asserts:
    1. All gaps are individually within the 300ms grace period.
    2. Observed closed time accumulates: 0.6s + 0.5s + 0.5s = 1.6s (> 1.5s threshold).
    3. Drowsiness alarm triggers on the 5th frame of the 3rd closed segment (trigger delay ~2.0s).
    """
    dt = 0.1
    fps = 10.0
    steps: list[ScenarioStep] = []
    t = 0.0

    # Prefix: 1.0s open
    for _ in range(10):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.35, note="open"))
        t += dt

    closure_onset = round(t, 4)

    # Segment 1: 0.6s closed (7 frames spans 0.6s)
    for _ in range(7):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.05, note="seg1_closed"))
        t += dt

    # Gap 1: 0.1s tracking loss (1 frame)
    steps.append(ScenarioStep(timestamp=round(t, 4), has_face=False, ear=0.0, note="gap1"))
    t += dt

    # Segment 2: 0.5s closed (6 frames spans 0.5s)
    for _ in range(6):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.05, note="seg2_closed"))
        t += dt

    # Gap 2: 0.1s tracking loss (1 frame)
    steps.append(ScenarioStep(timestamp=round(t, 4), has_face=False, ear=0.0, note="gap2"))
    t += dt

    # Segment 3: 0.5s closed (6 frames spans 0.5s)
    for _ in range(6):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.05, note="seg3_closed"))
        t += dt

    res = run_scenario("e_repeated_gaps", steps, stimulus_onset_time=closure_onset)
    assert res.alarm_triggered is True, "Repeated short gaps <= 300ms failed to preserve closure continuity."
    assert res.trigger_delay_sec == pytest.approx(2.0, abs=0.05), (
        f"Expected trigger delay ~2.0s (1.5s closed + 2 x 0.1s gaps + 0.3s discretization), got {res.trigger_delay_sec}s."
    )


def test_scenario_e_normal_blink_interrupted_by_short_tracking_loss():
    """
    Scenario E4: Normal blink interrupted by brief tracking loss.
    Sequence: Open -> Closed for 0.1s (2 frames @ 10 FPS) -> 0.1s tracking loss -> Eyes open.
    Asserts:
    1. Blink is recognized and blink_count increments by 1.
    2. Zero false drowsiness alarms triggered.
    3. Continuity tracking fields reset cleanly upon eye reopening.
    """
    dt = 0.1
    steps = [
        ScenarioStep(timestamp=0.0, has_face=True, ear=0.35, note="open_init"),
        ScenarioStep(timestamp=0.1, has_face=True, ear=0.05, note="blink_closed_1"),
        ScenarioStep(timestamp=0.2, has_face=True, ear=0.05, note="blink_closed_2"),
        ScenarioStep(timestamp=0.3, has_face=False, ear=0.0, note="tracking_gap"),
        ScenarioStep(timestamp=0.4, has_face=True, ear=0.35, note="reopened"),
        ScenarioStep(timestamp=0.5, has_face=True, ear=0.35, note="open_steady"),
    ]
    res = run_scenario("e_blink_gap", steps)
    assert res.total_blinks == 1, (
        f"Expected exactly 1 blink detected for blink interrupted by 100ms gap, got {res.total_blinks}."
    )
    assert res.alarm_triggered is False, "Normal blink interrupted by gap should not trigger alarm."


def test_scenario_e_alarm_recovery_hysteresis_preserved_across_tracking_loss():
    """
    Scenario E5: Alarm recovery hysteresis across tracking loss.
    Sequence:
    1. 2.0s continuous closure triggers alarm.
    2. Tracking is lost for 0.5s.
       Alarm MUST remain TRIGGERED (cannot clear while tracking is lost).
    3. Eyes return open for 0.5s (< 1.0s recovery duration).
       Alarm MUST remain TRIGGERED (verifying alertness).
    4. Eyes remain open for another 0.6s (total 1.1s continuous open eyes).
       Alarm clears to OFF and state to NORMAL.
    """
    dt = 0.1
    steps: list[ScenarioStep] = []
    t = 0.0

    # 1. 2.0s closure (21 frames)
    for _ in range(21):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.05, note="closure"))
        t += dt

    # 2. 0.5s tracking loss (5 frames)
    for _ in range(5):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=False, ear=0.0, note="tracking_lost"))
        t += dt

    # 3. 0.5s open recovery (5 frames, open duration = 0.4s < 1.0s)
    for _ in range(5):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.35, note="recovery_incomplete"))
        t += dt

    # 4. Another 0.6s open recovery (6 frames, continuous open = 1.0s)
    for _ in range(6):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.35, note="recovery_complete"))
        t += dt

    res = run_scenario("e_recovery_across_gap", steps, stimulus_onset_time=0.0, recovery_onset_time=2.6)
    assert res.alarm_triggered is True, "Alarm failed to trigger from 2.0s closure."
    assert res.recovery_cleared_timestamp == pytest.approx(3.6, abs=0.05), (
        f"Expected recovery at t=3.6s (1.0s continuous open after re-acquisition), got {res.recovery_cleared_timestamp}."
    )


def test_scenario_e_missing_face_alone_does_not_trigger_alarm():
    """
    Scenario E6: Missing face alone must never trigger a drowsiness alarm.
    Sequence: 1.0s normal open driving -> 2.0s missing face.
    Asserts:
    1. Zero alarm frames throughout the missing-face duration.
    2. Tracking state transitions to LOST after timeout (0.8s).
    3. Detection state reaches TRACKING_LOST.
    """
    dt = 0.1
    steps: list[ScenarioStep] = []
    t = 0.0

    for _ in range(10):  # 1.0s open
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.35, note="open"))
        t += dt

    for _ in range(20):  # 2.0s face lost
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=False, ear=0.0, note="lost"))
        t += dt

    res = run_scenario("e_missing_face_alone", steps)
    assert res.alarm_triggered is False, "Missing face alone triggered an unauthorized drowsiness alarm."
    assert res.total_alarm_frames == 0, "Alarm frames reported during face loss."
    assert res.timeline[-1].tracking_state == TrackingState.LOST.value
    assert res.timeline[-1].detection_state == DetectionState.TRACKING_LOST.value


def test_scenario_f_perclos_observation_gaps_and_integration():
    """
    Scenario F: PERCLOS time integration across gaps.
    60s total sequence: 10s closed -> 10s lost -> 10s open -> 10s lost -> 10s closed -> 10s lost.
    Asserts:
    1. Valid observation time is 30.0s (>= 20.0s min_observation_sec), yielding status VALID.
    2. Closed time is 20.0s, yielding PERCLOS = 20 / 30 = 0.667 (66.7%).
    3. Confirms that PERCLOS integrates valid tracked frames, not total elapsed wall clock.
    """
    steps = build_perclos_gap_steps(fps=10.0)
    res = run_scenario("f_perclos_gaps", steps)

    assert res.final_perclos_valid is True, "Expected PERCLOS to be VALID with 30s of observation."
    assert res.final_perclos == pytest.approx(20.0 / 30.0, abs=0.015), (
        f"Expected PERCLOS ~66.7% (20s closed / 30s valid), got {res.final_perclos * 100:.1f}%."
    )


def test_scenario_g_calibration_quality_gates():
    """
    Scenario G: Baseline calibration validation gates and threshold clamping.
    Asserts:
    1. Normal open EAR (0.35): valid, threshold = 0.245 (0.35 * 0.70).
    2. Borderline low (0.22): valid, threshold clamped to min floor 0.180 (0.22 * 0.70 = 0.154 -> 0.180).
    3. Borderline high (0.48): valid, threshold clamped to max ceiling 0.320 (0.48 * 0.70 = 0.336 -> 0.320).
    4. Outlier open EAR (< 0.20): rejected with EXTREME_EAR_OUTLIER, default 0.250 restored.
    5. Low tracking ratio (50%): rejected with LOW_TRACKING_RATIO, default 0.250 restored.
    6. Suspected drowsiness (> 0.75s closure): rejected with SUSPECTED_DROWSINESS, default 0.250 restored.
    """
    # G1: Normal open eye
    g1 = evaluate_calibration_scenario("g1", [(i * 0.1, 0.35, True, True) for i in range(120)])
    assert g1.is_valid is True
    assert g1.personalized_closure_threshold == pytest.approx(0.245, abs=1e-3)

    # G2: Borderline low (clamped to 0.18)
    g2 = evaluate_calibration_scenario("g2", [(i * 0.1, 0.22, True, True) for i in range(120)])
    assert g2.is_valid is True
    assert g2.personalized_closure_threshold == 0.18

    # G3: Borderline high (clamped to 0.32)
    g3 = evaluate_calibration_scenario("g3", [(i * 0.1, 0.48, True, True) for i in range(120)])
    assert g3.is_valid is True
    assert g3.personalized_closure_threshold == 0.32

    # G4: Outlier EAR < 0.20
    g4 = evaluate_calibration_scenario("g4", [(i * 0.1, 0.15, True, True) for i in range(120)])
    assert g4.is_valid is False
    assert g4.failure_reason == CalibrationFailureReason.EXTREME_EAR_OUTLIER
    assert g4.personalized_closure_threshold == 0.25

    # G5: Low tracking ratio (50%)
    g5 = evaluate_calibration_scenario("g5", [(i * 0.1, 0.35 if (i % 2 == 0) else 0.0, (i % 2 == 0), (i % 2 == 0)) for i in range(120)])
    assert g5.is_valid is False
    assert g5.failure_reason == CalibrationFailureReason.LOW_TRACKING_RATIO
    assert g5.personalized_closure_threshold == 0.25

    # G6: Suspected drowsiness during calibration
    g6_samples = [(i * 0.1, 0.05 if (4.0 <= i * 0.1 <= 5.0) else 0.35, True, True) for i in range(120)]
    g6 = evaluate_calibration_scenario("g6", g6_samples)
    assert g6.is_valid is False
    assert g6.failure_reason == CalibrationFailureReason.SUSPECTED_DROWSINESS
    assert g6.personalized_closure_threshold == 0.25
