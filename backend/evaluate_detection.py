"""
AntiRHys Detection & Evaluation Benchmark Harness
================================================
Provides automated, reproducible evaluation of the AntiRHys detection engine
across synthetic driver behavioral scenarios:
- Scenario A: Normal attentive behavior (blinking vs false alarms)
- Scenario B: Prolonged eye closure (trigger latency across durations 1.0s, 1.5s, 2.0s, 3.0s)
- Scenario C: Alarm recovery hysteresis & interrupted recovery
- Scenario D: Boundary EAR noise oscillation & state transition dynamics
- Scenario E: Brief tracking loss during eye closure
- Scenario F: PERCLOS time integration & gap handling
- Scenario G: Baseline calibration gating & threshold clamping

Usage:
    python evaluate_detection.py [--export [filename.json]]
"""

from dataclasses import dataclass, field, asdict
import json
import math
import os
import sys
import time
from typing import List, Tuple, Dict, Any, Optional

from detector import (
    DrowsinessDetector,
    DetectionConfig,
    DetectionState,
    AlarmState,
    TrackingState,
    EyeState,
    PerclosStatus,
    calculate_ear,
)
from baseline_calibrator import (
    BaselineCalibrator,
    CalibrationConfig,
    CalibrationState,
    CalibrationFailureReason,
    DriverBaseline,
)


def make_eye_landmarks(target_ear: float) -> List[Tuple[float, float]]:
    """
    Generate 6 2D eye landmark points with horizontal width 20.0 px and
    vertical opening h = 20.0 * target_ear.
    Guarantees calculate_ear() produces the exact target_ear value.
    """
    h = 20.0 * max(0.0, float(target_ear))
    return [
        (10.0, 50.0),
        (16.0, 50.0 - h / 2.0),
        (24.0, 50.0 - h / 2.0),
        (30.0, 50.0),
        (24.0, 50.0 + h / 2.0),
        (16.0, 50.0 + h / 2.0),
    ]


# Pre-defined landmark templates for nominal states
OPEN_EYE = make_eye_landmarks(0.35)
CLOSED_EYE = make_eye_landmarks(0.05)


@dataclass
class ScenarioStep:
    """A discrete observation step within an evaluation scenario."""
    timestamp: float
    has_face: bool
    ear: float
    note: str = ""


@dataclass
class TimelineRecord:
    """Recorded detector output at a single step."""
    timestamp: float
    has_face: bool
    ear: float
    eye_state: str
    tracking_state: str
    detection_state: str
    alarm_state: str
    closure_duration: float
    blink_count: int
    perclos: float
    perclos_valid: bool
    perclos_status: str


@dataclass
class ScenarioResult:
    """Structured evaluation output for a synthetic scenario."""
    scenario_name: str
    total_frames: int
    total_duration_sec: float
    alarm_triggered: bool
    first_alarm_timestamp: Optional[float]
    trigger_delay_sec: Optional[float]
    recovery_cleared_timestamp: Optional[float]
    recovery_delay_sec: Optional[float]
    total_alarm_frames: int
    total_blinks: int
    final_perclos: float
    final_perclos_valid: bool
    final_perclos_status: str
    timeline: List[TimelineRecord] = field(default_factory=list)

    def to_dict(self, include_timeline: bool = False) -> Dict[str, Any]:
        data = {
            "scenario_name": self.scenario_name,
            "total_frames": self.total_frames,
            "total_duration_sec": round(self.total_duration_sec, 3),
            "alarm_triggered": self.alarm_triggered,
            "first_alarm_timestamp": (
                round(self.first_alarm_timestamp, 3)
                if self.first_alarm_timestamp is not None
                else None
            ),
            "trigger_delay_sec": (
                round(self.trigger_delay_sec, 3)
                if self.trigger_delay_sec is not None
                else None
            ),
            "recovery_cleared_timestamp": (
                round(self.recovery_cleared_timestamp, 3)
                if self.recovery_cleared_timestamp is not None
                else None
            ),
            "recovery_delay_sec": (
                round(self.recovery_delay_sec, 3)
                if self.recovery_delay_sec is not None
                else None
            ),
            "total_alarm_frames": self.total_alarm_frames,
            "total_blinks": self.total_blinks,
            "final_perclos": round(self.final_perclos, 4),
            "final_perclos_valid": self.final_perclos_valid,
            "final_perclos_status": self.final_perclos_status,
        }
        if include_timeline:
            data["timeline"] = [asdict(t) for t in self.timeline]
        return data


def run_scenario(
    name: str,
    steps: List[ScenarioStep],
    config: Optional[DetectionConfig] = None,
    stimulus_onset_time: Optional[float] = None,
    recovery_onset_time: Optional[float] = None,
) -> ScenarioResult:
    """
    Execute a named synthetic scenario through the production DrowsinessDetector.
    Computes exact trigger delay and recovery delay relative to specified stimulus onsets.
    """
    detector = DrowsinessDetector(session_id=f"eval_{name}", config=config or DetectionConfig())
    timeline: List[TimelineRecord] = []

    alarm_triggered = False
    first_alarm_time: Optional[float] = None
    alarm_frames = 0
    recovery_cleared_time: Optional[float] = None

    for step in steps:
        if step.has_face:
            pts = make_eye_landmarks(step.ear)
            res = detector.process_frame_landmarks(
                has_face=True,
                left_eye_points=pts,
                right_eye_points=pts,
                now=step.timestamp,
            )
        else:
            res = detector.process_frame_landmarks(
                has_face=False,
                now=step.timestamp,
            )

        if res.alarm_state == AlarmState.TRIGGERED:
            alarm_frames += 1
            if not alarm_triggered:
                alarm_triggered = True
                first_alarm_time = step.timestamp
        elif alarm_triggered and res.alarm_state == AlarmState.OFF and recovery_cleared_time is None:
            recovery_cleared_time = step.timestamp

        timeline.append(
            TimelineRecord(
                timestamp=step.timestamp,
                has_face=step.has_face,
                ear=step.ear,
                eye_state=res.eye_state.value,
                tracking_state=res.tracking_state.value,
                detection_state=res.detection_state.value,
                alarm_state=res.alarm_state.value,
                closure_duration=res.closure_duration,
                blink_count=res.blink_count,
                perclos=res.perclos,
                perclos_valid=res.perclos_valid,
                perclos_status=res.perclos_status,
            )
        )

    total_dur = (steps[-1].timestamp - steps[0].timestamp) if len(steps) > 1 else 0.0

    trigger_delay = None
    if first_alarm_time is not None and stimulus_onset_time is not None:
        trigger_delay = first_alarm_time - stimulus_onset_time

    recovery_delay = None
    if recovery_cleared_time is not None and recovery_onset_time is not None:
        recovery_delay = recovery_cleared_time - recovery_onset_time

    last_rec = timeline[-1] if timeline else None

    return ScenarioResult(
        scenario_name=name,
        total_frames=len(steps),
        total_duration_sec=total_dur,
        alarm_triggered=alarm_triggered,
        first_alarm_timestamp=first_alarm_time,
        trigger_delay_sec=trigger_delay,
        recovery_cleared_timestamp=recovery_cleared_time,
        recovery_delay_sec=recovery_delay,
        total_alarm_frames=alarm_frames,
        total_blinks=last_rec.blink_count if last_rec else 0,
        final_perclos=last_rec.perclos if last_rec else 0.0,
        final_perclos_valid=last_rec.perclos_valid if last_rec else False,
        final_perclos_status=last_rec.perclos_status if last_rec else "UNAVAILABLE",
        timeline=timeline,
    )


# ------------------------------------------------------------------------------
# Synthetic Scenario Builders
# ------------------------------------------------------------------------------

def build_normal_attentive_steps(
    duration_sec: float = 60.0,
    fps: float = 10.0,
    blink_interval: float = 4.0,
    blink_duration: float = 0.20,
    open_ear: float = 0.35,
    closed_ear: float = 0.05,
) -> List[ScenarioStep]:
    """Generate attentive driving with periodic blinks."""
    total_frames = int(duration_sec * fps)
    dt = 1.0 / fps
    steps: List[ScenarioStep] = []
    for i in range(total_frames):
        t = round(i * dt, 4)
        in_blink = (t % blink_interval) < blink_duration
        ear = closed_ear if in_blink else open_ear
        steps.append(ScenarioStep(timestamp=t, has_face=True, ear=ear, note="blink" if in_blink else "open"))
    return steps


def build_prolonged_closure_steps(
    closure_duration: float,
    fps: float = 10.0,
    open_prefix: float = 2.0,
    open_suffix: float = 2.0,
    open_ear: float = 0.35,
    closed_ear: float = 0.05,
) -> Tuple[List[ScenarioStep], float]:
    """
    Generate open eyes -> continuous closure -> open eyes.
    Returns (steps, stimulus_onset_time).
    """
    dt = 1.0 / fps
    steps: List[ScenarioStep] = []
    t = 0.0

    # 1. Open prefix
    prefix_frames = int(open_prefix * fps)
    for _ in range(prefix_frames):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=open_ear, note="prefix_open"))
        t += dt

    stimulus_onset = round(t, 4)

    # 2. Closure phase
    closure_frames = int(round(closure_duration * fps))
    for _ in range(closure_frames):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=closed_ear, note="closed"))
        t += dt

    # 3. Open suffix
    suffix_frames = int(open_suffix * fps)
    for _ in range(suffix_frames):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=open_ear, note="suffix_open"))
        t += dt

    return steps, stimulus_onset


def build_recovery_hysteresis_steps(
    closure_duration: float = 2.0,
    open_recovery_duration: float = 2.5,
    interrupt_after_sec: Optional[float] = None,
    fps: float = 10.0,
) -> Tuple[List[ScenarioStep], float, float]:
    """
    Generate closure triggering alarm, followed by open-eye recovery.
    If interrupt_after_sec is provided, eyes close again after that duration.
    Returns (steps, closure_onset, reopen_onset).
    """
    dt = 1.0 / fps
    steps: List[ScenarioStep] = []
    t = 0.0

    # 1. Open prefix (1.0s)
    for _ in range(int(1.0 * fps)):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.35, note="prefix"))
        t += dt

    closure_onset = round(t, 4)

    # 2. Prolonged closure (2.0s -> triggers alarm at 1.5s)
    for _ in range(int(round(closure_duration * fps))):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.05, note="closed_alarm"))
        t += dt

    reopen_onset = round(t, 4)

    if interrupt_after_sec is not None:
        # Reopen briefly
        reopen_frames = int(round(interrupt_after_sec * fps))
        for _ in range(reopen_frames):
            steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.35, note="brief_open"))
            t += dt
        # Close again (interruption)
        for _ in range(int(1.5 * fps)):
            steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.05, note="interrupted_closure"))
            t += dt
    else:
        # Full recovery phase
        for _ in range(int(round(open_recovery_duration * fps))):
            steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.35, note="recovery_open"))
            t += dt

    return steps, closure_onset, reopen_onset


def build_boundary_noise_steps(
    duration_sec: float = 5.0,
    fps: float = 10.0,
    low_ear: float = 0.245,
    high_ear: float = 0.255,
) -> List[ScenarioStep]:
    """Generate alternating EAR values oscillating around the 0.250 threshold."""
    total_frames = int(duration_sec * fps)
    dt = 1.0 / fps
    steps: List[ScenarioStep] = []
    for i in range(total_frames):
        t = round(i * dt, 4)
        ear = low_ear if (i % 2 == 0) else high_ear
        steps.append(ScenarioStep(timestamp=t, has_face=True, ear=ear, note=f"oscillate_{ear}"))
    return steps


def build_tracking_loss_during_closure_steps(
    pre_loss_closure_sec: float = 1.0,
    loss_duration_sec: float = 0.2,
    post_loss_closure_sec: float = 0.6,
    fps: float = 10.0,
) -> Tuple[List[ScenarioStep], float]:
    """
    Simulate closure -> brief tracking loss -> closure.
    Discrete sample intervals span [t0, t0 + dur], i.e., int(round(dur * fps)) + 1 frames,
    so that observed elapsed time exactly equals requested duration.
    Returns (steps, initial_closure_onset).
    """
    dt = 1.0 / fps
    steps: List[ScenarioStep] = []
    t = 0.0

    # Prefix (1.0s open)
    for _ in range(int(round(1.0 * fps))):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.35, note="open"))
        t += dt

    closure_onset = round(t, 4)

    # Initial closure: span pre_loss_closure_sec
    pre_frames = int(round(pre_loss_closure_sec * fps)) + 1
    for _ in range(pre_frames):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.05, note="pre_loss_closed"))
        t += dt

    # Tracking lost: loss_duration_sec gap
    loss_frames = max(1, int(round(loss_duration_sec * fps)))
    for _ in range(loss_frames):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=False, ear=0.0, note="face_lost"))
        t += dt

    # Post-loss closure: span post_loss_closure_sec
    post_frames = int(round(post_loss_closure_sec * fps)) + 1
    for _ in range(post_frames):
        steps.append(ScenarioStep(timestamp=round(t, 4), has_face=True, ear=0.05, note="post_loss_closed"))
        t += dt

    return steps, closure_onset


def build_perclos_gap_steps(fps: float = 10.0) -> List[ScenarioStep]:
    """
    Generate 60-second scenario with alternating tracking gaps:
    10s closed -> 10s lost -> 10s open -> 10s lost -> 10s closed -> 10s lost.
    Total wall clock: 60s. Valid observation: 30s. Total closed: 20s.
    Expected PERCLOS = 20 / 30 = 0.667 (66.7%).
    """
    dt = 1.0 / fps
    steps: List[ScenarioStep] = []
    t = 0.0
    segments = [
        (10.0, True, 0.05, "10s_closed"),
        (10.0, False, 0.0, "10s_lost"),
        (10.0, True, 0.35, "10s_open"),
        (10.0, False, 0.0, "10s_lost"),
        (10.0, True, 0.05, "10s_closed"),
        (10.0, False, 0.0, "10s_lost"),
    ]
    for dur, has_face, ear, tag in segments:
        frames = int(round(dur * fps))
        for _ in range(frames):
            steps.append(ScenarioStep(timestamp=round(t, 4), has_face=has_face, ear=ear, note=tag))
            t += dt
    return steps


# ------------------------------------------------------------------------------
# Calibration Evaluation Routine
# ------------------------------------------------------------------------------

def evaluate_calibration_scenario(
    driver_id: str,
    samples: List[Tuple[float, float, bool, bool]],  # (timestamp, ear, has_face, valid_geom)
    config: Optional[CalibrationConfig] = None,
) -> DriverBaseline:
    """Run calibration samples through BaselineCalibrator and return baseline."""
    calibrator = BaselineCalibrator(config=config or CalibrationConfig())
    start_t = samples[0][0] if samples else 0.0
    end_t = samples[-1][0] if samples else 12.0
    calibrator.start_calibration(driver_id=driver_id, start_timestamp=start_t)
    for ts, ear, face, geom in samples:
        calibrator.add_sample(timestamp=ts, ear=ear, face_detected=face, is_valid_geometry=geom)
    return calibrator.finalize(end_timestamp=end_t)


# ------------------------------------------------------------------------------
# Backward Compatibility Benchmarks
# ------------------------------------------------------------------------------

def run_benchmark_alert_stream(duration_sec: float = 60.0, fps: float = 10.0) -> Dict[str, Any]:
    steps = build_normal_attentive_steps(duration_sec=duration_sec, fps=fps)
    t_start = time.perf_counter()
    res = run_scenario("alert_stream", steps)
    bench_wall_time = time.perf_counter() - t_start
    return {
        "duration_sec": duration_sec,
        "total_frames": res.total_frames,
        "simulated_blinks_injected": int(duration_sec / 4.0),
        "blinks_detected": res.total_blinks,
        "false_alarms": res.total_alarm_frames,
        "benchmark_wall_sec": round(bench_wall_time, 4),
        "avg_frame_process_micros": round((bench_wall_time / res.total_frames) * 1_000_000, 2),
    }


def run_benchmark_drowsiness_episodes(episodes: int = 10, closure_duration: float = 2.5, fps: float = 10.0) -> Dict[str, Any]:
    latencies: List[float] = []
    recovery_times: List[float] = []
    detected_count = 0

    for ep in range(episodes):
        steps, closure_onset, reopen_onset = build_recovery_hysteresis_steps(
            closure_duration=closure_duration,
            open_recovery_duration=2.5,
            fps=fps
        )
        res = run_scenario(f"drowsy_ep_{ep}", steps, stimulus_onset_time=closure_onset, recovery_onset_time=reopen_onset)
        if res.alarm_triggered:
            detected_count += 1
            if res.trigger_delay_sec is not None:
                latencies.append(res.trigger_delay_sec)
        if res.recovery_delay_sec is not None:
            recovery_times.append(res.recovery_delay_sec)

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    avg_rec = sum(recovery_times) / len(recovery_times) if recovery_times else 0.0

    return {
        "episodes_tested": episodes,
        "episodes_detected": detected_count,
        "recall_rate": detected_count / episodes if episodes > 0 else 0.0,
        "avg_detection_latency_sec": round(avg_lat, 3),
        "configured_threshold_sec": 1.5,
        "avg_recovery_latency_sec": round(avg_rec, 3),
        "configured_recovery_sec": 1.0,
    }


def run_benchmark_tracking_loss(fps: float = 10.0) -> Dict[str, Any]:
    steps: List[ScenarioStep] = []
    dt = 1.0 / fps
    for i in range(int(1.0 * fps)):
        steps.append(ScenarioStep(timestamp=round(i * dt, 4), has_face=True, ear=0.35))
    for i in range(int(2.0 * fps)):
        steps.append(ScenarioStep(timestamp=round(1.0 + i * dt, 4), has_face=False, ear=0.0))

    res = run_scenario("tracking_loss_bench", steps)
    lost_frames = sum(1 for t in res.timeline if t.detection_state == "TRACKING_LOST")
    alert_during_loss = sum(1 for t in res.timeline if not t.has_face and t.detection_state == "NORMAL")

    return {
        "falsely_reported_alert_during_loss": alert_during_loss,
        "tracking_lost_verified": lost_frames > 0,
        "tracking_timeout_configured_sec": 0.8,
    }


# ------------------------------------------------------------------------------
# Full Harness Runner CLI
# ------------------------------------------------------------------------------

def run_all_evaluation_scenarios() -> Dict[str, Any]:
    """Execute all defined synthetic evaluation scenarios and return structured report."""
    results: Dict[str, Any] = {}

    # Scenario A: Attentive driving (60s @ 10 FPS)
    s_a = build_normal_attentive_steps(duration_sec=60.0, fps=10.0)
    res_a = run_scenario("A_normal_attentive", s_a)
    results["scenario_A_normal_attentive"] = res_a.to_dict()

    # Scenario B: Prolonged closures (1.0s, 1.5s, 2.0s, 3.0s)
    b_durations = [1.0, 1.5, 2.0, 3.0]
    b_results = {}
    for dur in b_durations:
        s_b, onset = build_prolonged_closure_steps(closure_duration=dur, fps=10.0)
        res_b = run_scenario(f"B_closure_{dur}s", s_b, stimulus_onset_time=onset)
        b_results[f"{dur}s"] = res_b.to_dict()
    results["scenario_B_prolonged_closures"] = b_results

    # Scenario C: Recovery Hysteresis (nominal & interrupted)
    s_c_nom, c_onset, c_reopen = build_recovery_hysteresis_steps(closure_duration=2.0, open_recovery_duration=2.5, fps=10.0)
    res_c_nom = run_scenario("C_recovery_nominal", s_c_nom, stimulus_onset_time=c_onset, recovery_onset_time=c_reopen)

    s_c_int, c_int_onset, _ = build_recovery_hysteresis_steps(closure_duration=2.0, interrupt_after_sec=0.5, fps=10.0)
    res_c_int = run_scenario("C_recovery_interrupted", s_c_int, stimulus_onset_time=c_int_onset)
    results["scenario_C_recovery_hysteresis"] = {
        "nominal": res_c_nom.to_dict(),
        "interrupted": res_c_int.to_dict(),
    }

    # Scenario D: Boundary EAR noise (0.245 <-> 0.255 @ 10 FPS for 5s)
    s_d = build_boundary_noise_steps(duration_sec=5.0, fps=10.0)
    res_d = run_scenario("D_boundary_noise", s_d)
    results["scenario_D_boundary_noise"] = res_d.to_dict()

    # Scenario E: Tracking loss during closure (0.1s, 0.2s, 0.4s)
    e_losses = [0.1, 0.2, 0.4]
    e_results = {}
    for loss_dur in e_losses:
        s_e, onset = build_tracking_loss_during_closure_steps(pre_loss_closure_sec=1.0, loss_duration_sec=loss_dur, post_loss_closure_sec=0.6, fps=10.0)
        res_e = run_scenario(f"E_loss_{loss_dur}s", s_e, stimulus_onset_time=onset)
        e_results[f"{loss_dur}s"] = res_e.to_dict()
    results["scenario_E_tracking_loss_during_closure"] = e_results

    # Scenario F: PERCLOS observation gaps
    s_f = build_perclos_gap_steps(fps=10.0)
    res_f = run_scenario("F_perclos_observation_gaps", s_f)
    results["scenario_F_perclos_gaps"] = res_f.to_dict()

    # Scenario G: Calibration quality
    # G1: Normal open eye (12s @ 0.35)
    g1_samples = [(i * 0.1, 0.35, True, True) for i in range(120)]
    b_g1 = evaluate_calibration_scenario("g1_normal", g1_samples)

    # G2: Borderline low (12s @ 0.22)
    g2_samples = [(i * 0.1, 0.22, True, True) for i in range(120)]
    b_g2 = evaluate_calibration_scenario("g2_low", g2_samples)

    # G3: Borderline high (12s @ 0.48)
    g3_samples = [(i * 0.1, 0.48, True, True) for i in range(120)]
    b_g3 = evaluate_calibration_scenario("g3_high", g3_samples)

    # G4: Outlier < 0.20 (12s @ 0.15)
    g4_samples = [(i * 0.1, 0.15, True, True) for i in range(120)]
    b_g4 = evaluate_calibration_scenario("g4_outlier", g4_samples)

    # G5: Low tracking stability (50% loss)
    g5_samples = [(i * 0.1, 0.35 if (i % 2 == 0) else 0.0, (i % 2 == 0), (i % 2 == 0)) for i in range(120)]
    b_g5 = evaluate_calibration_scenario("g5_low_tracking", g5_samples)

    # G6: Drowsy calibration (closure > 0.75s)
    g6_samples = []
    for i in range(120):
        t = i * 0.1
        ear = 0.05 if (4.0 <= t <= 5.0) else 0.35  # 1.0s closure
        g6_samples.append((t, ear, True, True))
    b_g6 = evaluate_calibration_scenario("g6_drowsy", g6_samples)

    results["scenario_G_calibration_quality"] = {
        "G1_normal": b_g1.to_dict(),
        "G2_borderline_low_clamped": b_g2.to_dict(),
        "G3_borderline_high_clamped": b_g3.to_dict(),
        "G4_extreme_outlier_rejected": b_g4.to_dict(),
        "G5_low_tracking_rejected": b_g5.to_dict(),
        "G6_drowsy_rejected": b_g6.to_dict(),
    }

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("AntiRHys Reproducible Detection Evaluation Suite")
    print("=" * 70)

    eval_data = run_all_evaluation_scenarios()

    export_path = None
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export_path = sys.argv[2] if len(sys.argv) > 2 else "detection_eval_results.json"
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2)
        print(f"[Export] Saved structured evaluation results to {export_path}")

    # Print summary table
    print("\n[Scenario Summary]")
    print(f"- Scenario A (Normal Attentive 60s): Alarms={eval_data['scenario_A_normal_attentive']['total_alarm_frames']}, Blinks={eval_data['scenario_A_normal_attentive']['total_blinks']}")

    print("\n- Scenario B (Prolonged Closures):")
    for dur, data in eval_data["scenario_B_prolonged_closures"].items():
        trig = data["alarm_triggered"]
        delay = f"{data['trigger_delay_sec']}s" if data['trigger_delay_sec'] is not None else "N/A"
        print(f"  * {dur}: Triggered={trig}, Trigger Delay={delay}")

    print("\n- Scenario C (Recovery Hysteresis):")
    nom = eval_data["scenario_C_recovery_hysteresis"]["nominal"]
    print(f"  * Nominal Recovery: Triggered={nom['alarm_triggered']}, Cleared In={nom['recovery_delay_sec']}s (target: 1.0s)")
    inter = eval_data["scenario_C_recovery_hysteresis"]["interrupted"]
    print(f"  * Interrupted Recovery: Cleared={inter['recovery_cleared_timestamp'] is not None}, Alarm Frames={inter['total_alarm_frames']}")

    print("\n- Scenario D (Boundary Noise 0.245 <-> 0.255 @ 10 Hz):")
    d_data = eval_data["scenario_D_boundary_noise"]
    print(f"  * Alarm Triggered={d_data['alarm_triggered']}, Involuntary Blink Counter Inflation={d_data['total_blinks']} blinks")

    print("\n- Scenario E (Tracking Loss During 2.0s Total Closure):")
    for loss_dur, data in eval_data["scenario_E_tracking_loss_during_closure"].items():
        print(f"  * Gap={loss_dur}: Alarm Triggered={data['alarm_triggered']}, Trigger Delay={data['trigger_delay_sec']}")

    print("\n- Scenario F (PERCLOS Observation Gaps - 60s total / 30s tracked / 20s closed):")
    f_data = eval_data["scenario_F_perclos_gaps"]
    print(f"  * Final PERCLOS={f_data['final_perclos'] * 100:.1f}%, Status={f_data['final_perclos_status']}")

    print("\n- Scenario G (Calibration Quality & Gating):")
    for name, g_data in eval_data["scenario_G_calibration_quality"].items():
        valid = g_data["is_valid"]
        reason = g_data["failure_reason"]
        thresh = g_data["personalized_closure_threshold"]
        print(f"  * {name}: Valid={valid}, Reason={reason}, Personalized Threshold={thresh}")

    print("\n" + "=" * 70)
