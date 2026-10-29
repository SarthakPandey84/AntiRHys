"""
AntiRHys Detection Engine
=========================
Provides robust Eye Aspect Ratio (EAR) computation, geometric validation,
and a temporal state machine for driver drowsiness monitoring.

Key Safety & Reliability Principles:
1. Elapsed-time-based detection using monotonic clock (time.monotonic()), independent of frame rate.
2. Explicit distinction between:
   - Eye state: OPEN, CLOSED, UNKNOWN
   - Tracking state: ACTIVE, LOST, SEARCHING
   - System detection state: NORMAL, POSSIBLE_DROWSINESS, TRACKING_LOST, UNAVAILABLE
   - Alarm state: OFF, TRIGGERED
3. Missing or lost face tracking is explicitly flagged as TRACKING_LOST, NEVER as "Alert" or "Safe".
4. Configurable provisional thresholds (clearly documented as requiring real-world calibration).
5. Hysteresis to prevent high-frequency jitter/flicker from oscillating alarm states.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
import os
import time
from typing import List, Tuple, Optional, Dict, Any

from baseline_calibrator import (
    BaselineCalibrator,
    CalibrationConfig,
    CalibrationState,
    CalibrationFailureReason,
    DriverBaseline
)
from head_pose import (
    HEAD_POSE_LANDMARK_INDICES,
    HeadPoseEstimator,
    HeadPoseResult,
    PoseStatus,
)


class EyeState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class TrackingState(str, Enum):
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    SEARCHING = "SEARCHING"


class DetectionState(str, Enum):
    INITIALIZING = "INITIALIZING"
    NORMAL = "NORMAL"
    POSSIBLE_DROWSINESS = "POSSIBLE_DROWSINESS"
    TRACKING_LOST = "TRACKING_LOST"
    CAMERA_OR_PROCESSING_ERROR = "CAMERA_OR_PROCESSING_ERROR"
    UNAVAILABLE = "UNAVAILABLE"


class AlarmState(str, Enum):
    OFF = "OFF"
    TRIGGERED = "TRIGGERED"


class PerclosStatus(str, Enum):
    VALID = "VALID"
    CALCULATING = "CALCULATING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNAVAILABLE = "UNAVAILABLE"


# Standard MediaPipe 468/478 Face Mesh landmark indices for eyes
# Formatted as: [p1(lateral), p2(upper1), p3(upper2), p4(medial), p5(lower2), p6(lower1)]
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


@dataclass
class DetectionConfig:
    """
    Configurable parameters for drowsiness, eye tracking, and PERCLOS.

    NOTE: Default thresholds are PROVISIONAL engineering baselines and must
    not be considered certified or scientifically validated for production safety.
    Thresholds require individual and multi-subject calibration.
    """
    ear_threshold: float = 0.25
    # Consecutive closure duration (seconds) required before declaring suspected acute drowsiness
    closure_duration_threshold: float = 1.5
    # Minimum and maximum duration (seconds) for a normal voluntary/involuntary blink
    blink_min_duration: float = 0.08
    blink_max_duration: float = 0.45
    # Grace period (seconds) of missing face frames before declaring TRACKING_LOST
    tracking_lost_timeout: float = 0.8
    # Maximum tracking gap (seconds) across which eye closure continuity is preserved
    # (unobserved gap time is excluded from closure duration; only observed frames accumulate)
    tracking_gap_grace_sec: float = 0.30
    # Duration (seconds) eyes must remain continuously open to clear a drowsiness alert
    recovery_open_duration: float = 1.0
    # Minimum horizontal eye span to consider landmarks valid (prevents div by zero/degenerate points)
    min_eye_width: float = 1.0  # in pixels

    # -------------------------------------------------------------
    # PERCLOS (Percentage of Eye Closure) Parameters
    # -------------------------------------------------------------
    # Duration of the rolling observation window in seconds (standard: 60s)
    perclos_window_sec: float = 60.0
    # Minimum valid observation seconds required within rolling window to declare measurement valid
    perclos_min_observation_sec: float = 20.0
    # EAR threshold below which eyes are classified as closed for PERCLOS integration
    perclos_closure_ear_threshold: float = 0.25
    # Provisional cumulative eye closure warning threshold (e.g. 0.20 = 20% closure in window)
    perclos_warning_threshold: float = 0.20
    # Maximum gap between consecutive frames to treat observation as continuous
    max_sample_interval_sec: float = 0.5
    # Default nominal duration attributed to isolated samples or samples after tracking loss
    default_sample_duration_sec: float = 0.1
    # -------------------------------------------------------------
    # Head Pose Estimation configuration (observational research signal)
    # -------------------------------------------------------------
    enable_head_pose: bool = True
    head_nod_pitch_threshold: float = -20.0        # Provisional chin-down alert threshold (degrees)
    head_distraction_yaw_threshold: float = 25.0   # Provisional looking-away threshold (degrees)
    head_pose_max_reprojection_error: float = 15.0 # Max reprojection RMSE (pixels)

    @classmethod
    def from_env(cls) -> "DetectionConfig":
        """
        Load configuration with safe environment variable overrides.
        Falls back to provisional engineering baselines on missing or invalid env vars.
        """
        cfg = cls()
        overrides = {
            "EAR_THRESHOLD": ("ear_threshold", float),
            "CLOSURE_DURATION_THRESHOLD": ("closure_duration_threshold", float),
            "BLINK_MIN_DURATION": ("blink_min_duration", float),
            "BLINK_MAX_DURATION": ("blink_max_duration", float),
            "TRACKING_LOST_TIMEOUT": ("tracking_lost_timeout", float),
            "TRACKING_GAP_GRACE_SEC": ("tracking_gap_grace_sec", float),
            "RECOVERY_OPEN_DURATION": ("recovery_open_duration", float),
            "PERCLOS_WINDOW_SEC": ("perclos_window_sec", float),
            "PERCLOS_MIN_OBSERVATION_SEC": ("perclos_min_observation_sec", float),
            "PERCLOS_CLOSURE_EAR_THRESHOLD": ("perclos_closure_ear_threshold", float),
            "PERCLOS_WARNING_THRESHOLD": ("perclos_warning_threshold", float),
            "MAX_SAMPLE_INTERVAL_SEC": ("max_sample_interval_sec", float),
            "DEFAULT_SAMPLE_DURATION_SEC": ("default_sample_duration_sec", float),
            "HEAD_NOD_PITCH_THRESHOLD": ("head_nod_pitch_threshold", float),
            "HEAD_DISTRACTION_YAW_THRESHOLD": ("head_distraction_yaw_threshold", float),
            "HEAD_POSE_MAX_REPROJECTION_ERROR": ("head_pose_max_reprojection_error", float),
        }
        for env_key, (attr_name, cast_fn) in overrides.items():
            raw_val = os.getenv(env_key)
            if raw_val is not None and raw_val.strip() != "":
                try:
                    setattr(cfg, attr_name, cast_fn(raw_val.strip()))
                except (ValueError, TypeError) as e:
                    print(f"[DetectionConfig] Warning: invalid env override for {env_key}='{raw_val}': {e}")
        return cfg


@dataclass
class EyeObservation:
    """Represents a discrete valid eye aperture observation with duration."""
    timestamp: float          # Monotonic timestamp of observation
    duration: float           # Effective duration in seconds
    is_closed: bool           # True if EAR < perclos_closure_ear_threshold
    ear: float                # Instantaneous EAR value


class PerclosCalculator:
    """
    Calculates PERCLOS (Proportion of Eye Closure) over a configurable rolling time window.

    Formula:
        PERCLOS = (Total valid closed-eye time in window) / (Total valid observation time in window)

    Key Reliability Guarantees:
    1. Timestamp Integration: Does not assume fixed frame rates; integrates elapsed monotonic time.
    2. Exclusion of Unknowns: Missing-face frames, invalid landmark geometries, and tracking gaps
       are strictly excluded from valid observation time (counted as neither open nor closed).
    3. Gap Bounding: Large gaps (network latency, tracking dropouts) are not credited as valid time.
    4. Measurement Validity: Exposes status flags (CALCULATING, VALID, INSUFFICIENT_DATA) so
       incomplete or unrepresentative windows are not presented as definitive metrics.
    """

    def __init__(
        self,
        window_duration: float = 60.0,
        min_observation_duration: float = 20.0,
        closure_ear_threshold: float = 0.25,
        max_sample_interval: float = 0.5,
        default_sample_duration: float = 0.1
    ):
        self.window_duration = window_duration
        self.min_observation_duration = min_observation_duration
        self.closure_ear_threshold = closure_ear_threshold
        self.max_sample_interval = max_sample_interval
        self.default_sample_duration = default_sample_duration

        self.observations: deque[EyeObservation] = deque()
        self._last_valid_timestamp: Optional[float] = None
        self._first_observation_time: Optional[float] = None

    def reset(self) -> None:
        """Clear all historical observations and reset timers."""
        self.observations.clear()
        self._last_valid_timestamp = None
        self._first_observation_time = None

    def record_observation(self, ear: float, now: float) -> None:
        """
        Record a valid geometric eye observation at monotonic time `now`.
        Only called when face tracking and eye geometry are confirmed valid.
        """
        # Guard against non-finite or negative coordinates/timestamps
        if not (math.isfinite(now) and math.isfinite(ear)) or now < 0.0:
            return

        # Guard against duplicate, out-of-order, or non-monotonic timestamps
        if self._last_valid_timestamp is not None and now <= self._last_valid_timestamp:
            return

        if self._first_observation_time is None:
            self._first_observation_time = now

        if self._last_valid_timestamp is not None:
            dt = now - self._last_valid_timestamp
            if 0 < dt <= self.max_sample_interval:
                sample_duration = dt
            else:
                # Discontinuous gap (tracking resumed or delayed frame)
                sample_duration = self.default_sample_duration
        else:
            sample_duration = self.default_sample_duration

        self._last_valid_timestamp = now
        is_closed = ear < self.closure_ear_threshold

        self.observations.append(
            EyeObservation(
                timestamp=now,
                duration=sample_duration,
                is_closed=is_closed,
                ear=ear
            )
        )
        self._prune_window(now)

    def record_invalid_observation(self) -> None:
        """
        Mark that face tracking was lost or eye geometry was invalid.
        Breaks the timestamp continuity so the next valid frame does not credit
        the lost interval towards valid observation time.
        """
        self._last_valid_timestamp = None

    def _prune_window(self, now: float) -> None:
        cutoff = now - self.window_duration
        while self.observations and (self.observations[0].timestamp <= cutoff):
            self.observations.popleft()

    def compute_perclos(self, now: float) -> Tuple[float, bool, str, float, float]:
        """
        Compute PERCLOS over the rolling window [now - window_duration, now].

        Returns:
            Tuple of:
            - perclos (float): Proportion of eye closure (0.0 to 1.0)
            - is_valid (bool): True if valid observation time >= min_observation_duration
            - status (str): "VALID", "CALCULATING", "INSUFFICIENT_DATA"
            - total_valid_sec (float): Valid observation time in seconds
            - total_closed_sec (float): Closed-eye observation time in seconds
        """
        if not math.isfinite(now) or now < 0.0:
            return 0.0, False, PerclosStatus.UNAVAILABLE.value, 0.0, 0.0

        self._prune_window(now)

        if not self.observations:
            is_startup = (
                self._first_observation_time is None or
                (now - self._first_observation_time) < self.window_duration
            )
            status = PerclosStatus.CALCULATING.value if is_startup else PerclosStatus.INSUFFICIENT_DATA.value
            return 0.0, False, status, 0.0, 0.0

        cutoff = now - self.window_duration
        total_valid_sec = 0.0
        total_closed_sec = 0.0

        for obs in self.observations:
            # Segment of this observation that lies within [cutoff, now]
            start_time = max(cutoff, obs.timestamp - obs.duration)
            end_time = min(now, obs.timestamp)
            duration_in_window = max(0.0, end_time - start_time)
            total_valid_sec += duration_in_window
            if obs.is_closed:
                total_closed_sec += duration_in_window

        if total_valid_sec < 1e-4:
            is_startup = (
                self._first_observation_time is None or
                (now - self._first_observation_time) < self.window_duration
            )
            status = PerclosStatus.CALCULATING.value if is_startup else PerclosStatus.INSUFFICIENT_DATA.value
            return 0.0, False, status, 0.0, 0.0

        perclos = total_closed_sec / total_valid_sec
        perclos = max(0.0, min(1.0, perclos))

        if total_valid_sec >= self.min_observation_duration:
            status = PerclosStatus.VALID.value
            is_valid = True
        else:
            # Distinguish window startup from tracking loss
            is_startup = (
                self._first_observation_time is not None and
                (now - self._first_observation_time) < self.window_duration
            )
            status = PerclosStatus.CALCULATING.value if is_startup else PerclosStatus.INSUFFICIENT_DATA.value
            is_valid = False

        return perclos, is_valid, status, total_valid_sec, total_closed_sec


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute 2D Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_ear(eye_landmarks: List[Tuple[float, float]], min_width: float = 1.0) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) given 6 landmark points:
    p1, p2, p3, p4, p5, p6

    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)

    Returns:
        float: Computed EAR, or 0.0 if geometry is degenerate/invalid.
    """
    if len(eye_landmarks) != 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_landmarks

    # Validate coordinate numbers
    for pt in eye_landmarks:
        if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
            return 0.0

    dist_p1_p4 = euclidean_distance(p1, p4)
    if dist_p1_p4 < min_width:
        return 0.0

    dist_p2_p6 = euclidean_distance(p2, p6)
    dist_p3_p5 = euclidean_distance(p3, p5)

    ear = (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)
    return float(ear) if math.isfinite(ear) else 0.0


@dataclass
class DetectionResult:
    timestamp: float
    detection_state: DetectionState
    alarm_state: AlarmState
    tracking_state: TrackingState
    eye_state: EyeState
    ear: float
    left_ear: float
    right_ear: float
    blink_count: int
    closure_duration: float
    tracking_lost_duration: float
    session_id: str
    landmarks: List[Dict[str, float]] = field(default_factory=list)
    status_message: str = ""
    # PERCLOS metrics
    perclos: float = 0.0
    perclos_valid: bool = False
    perclos_status: str = "CALCULATING"
    perclos_window_duration: float = 0.0
    perclos_warning: bool = False
    # Baseline & Personalization metrics
    baseline_active: bool = False
    calibrated_ear_threshold: float = 0.25
    calibration_state: str = "IDLE"
    calibration_progress: float = 0.0
    calibration_failure_reason: str = "NONE"
    # 3D Head Pose Orientation metrics (observational research signal)
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    head_pose_valid: bool = False
    head_pose_status: str = "UNAVAILABLE"
    head_pose_reprojection_error: float = 0.0
    head_pose_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "detection_state": self.detection_state.value,
            "alarm_state": self.alarm_state.value,
            "tracking_state": self.tracking_state.value,
            "eye_state": self.eye_state.value,
            "ear": round(self.ear, 3),
            "left_ear": round(self.left_ear, 3),
            "right_ear": round(self.right_ear, 3),
            "blink_count": self.blink_count,
            "closure_duration": round(self.closure_duration, 2),
            "tracking_lost_duration": round(self.tracking_lost_duration, 2),
            "session_id": self.session_id,
            "landmarks": self.landmarks,
            "status_message": self.status_message,
            "perclos": round(self.perclos, 3),
            "perclos_valid": self.perclos_valid,
            "perclos_status": self.perclos_status,
            "perclos_window_duration": round(self.perclos_window_duration, 1),
            "perclos_warning": self.perclos_warning,
            "baseline_active": self.baseline_active,
            "calibrated_ear_threshold": round(self.calibrated_ear_threshold, 3),
            "calibration_state": self.calibration_state,
            "calibration_progress": round(self.calibration_progress, 2),
            "calibration_failure_reason": self.calibration_failure_reason,
            "pitch": round(self.pitch, 1),
            "yaw": round(self.yaw, 1),
            "roll": round(self.roll, 1),
            "head_pose_valid": self.head_pose_valid,
            "head_pose_status": self.head_pose_status,
            "head_pose_reprojection_error": round(self.head_pose_reprojection_error, 2),
            "head_pose_confidence": round(self.head_pose_confidence, 2),
        }


class DrowsinessDetector:
    """
    State machine for tracking eye closure, detecting potential drowsiness,
    and handling tracking availability with monotonic timing.
    """

    def __init__(self, session_id: str, config: Optional[DetectionConfig] = None):
        self.session_id = session_id
        self.config = config or DetectionConfig.from_env()

        # Operational state
        self.is_active = True
        self.tracking_state = TrackingState.SEARCHING
        self.eye_state = EyeState.UNKNOWN
        self.detection_state = DetectionState.NORMAL
        self.alarm_state = AlarmState.OFF
        self.blink_count = 0

        # Baseline & Personalization state
        self._default_ear_threshold: float = self.config.ear_threshold
        self._default_perclos_closure_threshold: float = self.config.perclos_closure_ear_threshold
        self.baseline: Optional[DriverBaseline] = None
        self.calibrator: Optional[BaselineCalibrator] = None
        self._last_frame_resolution: Optional[Tuple[int, int]] = None

        # Monotonic timestamps for transitions
        self._last_face_seen_time: Optional[float] = None
        self._face_lost_start_time: Optional[float] = None
        self._eyes_closed_start_time: Optional[float] = None
        self._eyes_open_start_time: Optional[float] = None
        self._last_processed_time: Optional[float] = None
        self._accumulated_closed_duration: float = 0.0
        self._last_closed_seen_time: Optional[float] = None

        # Integrated PERCLOS calculator
        self.perclos_calculator = PerclosCalculator(
            window_duration=self.config.perclos_window_sec,
            min_observation_duration=self.config.perclos_min_observation_sec,
            closure_ear_threshold=self.config.perclos_closure_ear_threshold,
            max_sample_interval=self.config.max_sample_interval_sec,
            default_sample_duration=self.config.default_sample_duration_sec
        )

        # 3D Head Pose Estimator (observational research signal)
        self.head_pose_estimator = HeadPoseEstimator(
            max_reprojection_error=self.config.head_pose_max_reprojection_error
        )

    def reset_session(self, session_id: Optional[str] = None):
        """Reset all temporal counters and timers for a fresh session."""
        if session_id:
            self.session_id = session_id
        self.is_active = True
        self.tracking_state = TrackingState.SEARCHING
        self.eye_state = EyeState.UNKNOWN
        self.detection_state = DetectionState.NORMAL
        self.alarm_state = AlarmState.OFF
        self.blink_count = 0
        self._last_face_seen_time = None
        self._face_lost_start_time = None
        self._eyes_closed_start_time = None
        self._eyes_open_start_time = None
        self._last_processed_time = None
        self._accumulated_closed_duration = 0.0
        self._last_closed_seen_time = None
        self._last_frame_resolution = None
        self.perclos_calculator.reset()
        if self.calibrator is not None:
            self.calibrator.reset()

    def _reset_baseline_thresholds(self) -> None:
        """Restore default uncalibrated thresholds and set baseline to None without resetting calibrator status."""
        self.baseline = None
        self.config.ear_threshold = self._default_ear_threshold
        self.config.perclos_closure_ear_threshold = self._default_perclos_closure_threshold
        self.perclos_calculator.closure_ear_threshold = self._default_perclos_closure_threshold

    def start_calibration(
        self,
        driver_id: str = "driver",
        config: Optional[CalibrationConfig] = None,
        start_timestamp: Optional[float] = None
    ) -> None:
        """Start or restart a baseline calibration window for this driver."""
        self._reset_baseline_thresholds()
        self.calibrator = BaselineCalibrator(config=config or CalibrationConfig())
        self.calibrator.start_calibration(driver_id=driver_id, start_timestamp=start_timestamp)

    def abort_calibration(self) -> None:
        """Cancel any active calibration."""
        self._reset_baseline_thresholds()
        if self.calibrator is not None:
            self.calibrator.abort()

    def finalize_calibration(self, now: Optional[float] = None) -> DriverBaseline:
        """Evaluate calibration samples, apply personalized threshold if valid, or fallback."""
        if self.calibrator is not None:
            baseline = self.calibrator.finalize(end_timestamp=now)
            self.apply_baseline(baseline)
            return baseline
        return DriverBaseline.default_fallback("anonymous")

    def _check_calibration_completion(self, now: float) -> None:
        """Finalize calibration if the collection window target duration has elapsed."""
        if self.calibrator and self.calibrator.state == CalibrationState.COLLECTING:
            if self.calibrator._start_time is not None:
                dur = now - self.calibrator._start_time
                if dur >= self.calibrator.config.target_duration_sec:
                    self.finalize_calibration(now)

    def apply_baseline(self, baseline: DriverBaseline) -> bool:
        """
        Safely consume personalized driver baseline into running detector and PERCLOS calculator.
        Returns True if valid baseline applied, False if rejected (falling back to default).
        """
        if baseline.is_valid:
            self.baseline = baseline
            self.config.ear_threshold = baseline.personalized_closure_threshold
            self.config.perclos_closure_ear_threshold = baseline.personalized_closure_threshold
            self.perclos_calculator.closure_ear_threshold = baseline.personalized_closure_threshold
            return True
        else:
            self._reset_baseline_thresholds()
            return False

    def clear_baseline(self) -> None:
        """Clear personalized baseline and restore default uncalibrated thresholds."""
        self._reset_baseline_thresholds()
        if self.calibrator is not None:
            self.calibrator.reset()

    def stop_session(self):
        """Mark session stopped and clear active state."""
        self.is_active = False
        self.detection_state = DetectionState.UNAVAILABLE
        self.alarm_state = AlarmState.OFF
        self.tracking_state = TrackingState.LOST
        self._accumulated_closed_duration = 0.0
        self._last_closed_seen_time = None
        self._eyes_closed_start_time = None
        self._eyes_open_start_time = None
        self.perclos_calculator.reset()

    @property
    def closure_start_time(self) -> Optional[float]:
        return self._eyes_closed_start_time

    @property
    def closure_duration(self) -> float:
        if self._eyes_closed_start_time is not None and self._last_processed_time is not None:
            return self._accumulated_closed_duration + max(0.0, self._last_processed_time - self._eyes_closed_start_time)
        return self._accumulated_closed_duration

    @property
    def perclos(self) -> float:
        t = self._last_processed_time if self._last_processed_time is not None else time.monotonic()
        p, _, _, _, _ = self.perclos_calculator.compute_perclos(t)
        return p

    @property
    def perclos_valid(self) -> bool:
        t = self._last_processed_time if self._last_processed_time is not None else time.monotonic()
        _, is_valid, _, _, _ = self.perclos_calculator.compute_perclos(t)
        return is_valid

    @property
    def perclos_status(self) -> str:
        t = self._last_processed_time if self._last_processed_time is not None else time.monotonic()
        _, _, status, _, _ = self.perclos_calculator.compute_perclos(t)
        return status

    @property
    def status_message(self) -> str:
        if self.detection_state == DetectionState.CAMERA_OR_PROCESSING_ERROR:
            return "Processing error encountered."
        if self.detection_state == DetectionState.POSSIBLE_DROWSINESS:
            return "Possible drowsiness detected!"
        if self.tracking_state == TrackingState.LOST:
            return "Face tracking lost."
        return "Monitoring normally."

    def process_frame_error(
        self,
        error_message: str,
        now: Optional[float] = None
    ) -> DetectionResult:
        """
        Record a camera or processing error (e.g. image decode failure, MediaPipe exception).
        Ensures error states are explicitly surfaced and cannot be mistaken for normal monitoring.
        """
        if now is None:
            now = time.monotonic()
        elif self._last_processed_time is not None and now < self._last_processed_time:
            now = self._last_processed_time

        self._last_processed_time = now
        self.eye_state = EyeState.UNKNOWN
        self.detection_state = DetectionState.CAMERA_OR_PROCESSING_ERROR
        self.tracking_state = TrackingState.LOST

        # Reset continuous closure timers so corrupt frames don't count towards closure duration
        self._eyes_closed_start_time = None
        self._eyes_open_start_time = None
        self._accumulated_closed_duration = 0.0
        self._last_closed_seen_time = None

        # Exclude corrupt frame from valid observation time
        self.perclos_calculator.record_invalid_observation()
        perclos, p_valid, p_status, _, _ = self.perclos_calculator.compute_perclos(now)

        calib_state = self.calibrator.state.value if self.calibrator else "IDLE"

        return DetectionResult(
            timestamp=now,
            detection_state=self.detection_state,
            alarm_state=self.alarm_state,
            tracking_state=self.tracking_state,
            eye_state=self.eye_state,
            ear=0.0,
            left_ear=0.0,
            right_ear=0.0,
            blink_count=self.blink_count,
            closure_duration=0.0,
            tracking_lost_duration=0.0,
            session_id=self.session_id,
            landmarks=[],
            status_message=f"Processing error: {error_message}",
            perclos=perclos,
            perclos_valid=p_valid,
            perclos_status=p_status,
            perclos_window_duration=self.config.perclos_window_sec,
            perclos_warning=False,
            baseline_active=(self.baseline is not None and self.baseline.is_valid),
            calibrated_ear_threshold=self.config.ear_threshold,
            calibration_state=calib_state,
            calibration_progress=0.0,
            calibration_failure_reason="NONE",
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            head_pose_valid=False,
            head_pose_status=PoseStatus.UNAVAILABLE.value,
            head_pose_reprojection_error=0.0,
            head_pose_confidence=0.0
        )

    def process_frame_landmarks(
        self,
        has_face: bool,
        left_eye_points: Optional[List[Tuple[float, float]]] = None,
        right_eye_points: Optional[List[Tuple[float, float]]] = None,
        normalized_landmarks: Optional[List[Dict[str, float]]] = None,
        now: Optional[float] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        head_pose_points: Optional[List[Tuple[float, float]]] = None
    ) -> DetectionResult:
        """
        Process observation for a single frame using monotonic time.

        Args:
            has_face: Whether FaceMesh detected a face in this frame.
            left_eye_points: 6 2D points for left eye in pixel coordinates.
            right_eye_points: 6 2D points for right eye in pixel coordinates.
            normalized_landmarks: Normalized coordinates for frontend visualization.
            now: Monotonic timestamp (defaults to time.monotonic()).
            frame_width: Optional width of input frame to track sensor changes.
            frame_height: Optional height of input frame to track sensor changes.
            head_pose_points: Optional 6 2D landmark points for 3D head pose estimation.
        """
        if now is None:
            now = time.monotonic()
        elif self._last_processed_time is not None and now < self._last_processed_time:
            now = self._last_processed_time

        self._last_processed_time = now

        # Detect camera stream / resolution change and invalidate personalized baseline
        if frame_width is not None and frame_height is not None and frame_width > 0 and frame_height > 0:
            current_res = (frame_width, frame_height)
            if self._last_frame_resolution is not None and self._last_frame_resolution != current_res:
                if self.baseline is not None and self.baseline.is_valid:
                    self.clear_baseline()
                if self.calibrator is not None and self.calibrator.state == CalibrationState.COLLECTING:
                    self.calibrator.abort()
                    self.calibrator.failure_reason = CalibrationFailureReason.LOW_TRACKING_RATIO
            self._last_frame_resolution = current_res

        if not self.is_active:
            return DetectionResult(
                timestamp=now,
                detection_state=DetectionState.UNAVAILABLE,
                alarm_state=AlarmState.OFF,
                tracking_state=TrackingState.LOST,
                eye_state=EyeState.UNKNOWN,
                ear=0.0,
                left_ear=0.0,
                right_ear=0.0,
                blink_count=self.blink_count,
                closure_duration=0.0,
                tracking_lost_duration=0.0,
                session_id=self.session_id,
                landmarks=[],
                status_message="Monitoring session stopped.",
                perclos=0.0,
                perclos_valid=False,
                perclos_status=PerclosStatus.UNAVAILABLE.value,
                perclos_window_duration=self.config.perclos_window_sec,
                perclos_warning=False,
                baseline_active=False,
                calibrated_ear_threshold=self.config.ear_threshold,
                calibration_state="IDLE",
                calibration_progress=0.0,
                calibration_failure_reason="NONE",
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                head_pose_valid=False,
                head_pose_status=PoseStatus.UNAVAILABLE.value,
                head_pose_reprojection_error=0.0,
                head_pose_confidence=0.0
            )

        # -------------------------------------------------------------
        # Branch 1: Face is NOT detected in current frame
        # -------------------------------------------------------------
        if not has_face or left_eye_points is None or right_eye_points is None:
            if self._face_lost_start_time is None:
                self._face_lost_start_time = now

            lost_duration = max(0.0, now - self._face_lost_start_time)
            self.eye_state = EyeState.UNKNOWN
            self.detection_state = DetectionState.TRACKING_LOST

            # Record sample in active calibrator (invalid observation)
            if self.calibrator and self.calibrator.state == CalibrationState.COLLECTING:
                self.calibrator.add_sample(timestamp=now, ear=0.0, face_detected=False, is_valid_geometry=False)
                self._check_calibration_completion(now)

            if lost_duration >= self.config.tracking_lost_timeout:
                self.tracking_state = TrackingState.LOST
                status_msg = f"Face tracking lost ({lost_duration:.1f}s)."
                if self.calibrator and self.calibrator.state == CalibrationState.COLLECTING:
                    self.calibrator.abort()
                    self.calibrator.failure_reason = CalibrationFailureReason.LOW_TRACKING_RATIO
            else:
                self.tracking_state = TrackingState.SEARCHING
                status_msg = "Face temporarily obscured or searching..."

            calib_state = self.calibrator.state.value if self.calibrator else "IDLE"
            calib_reason = self.calibrator.failure_reason.value if (self.calibrator and self.calibrator.state == CalibrationState.FAILED) else "NONE"
            calib_prog = 0.0
            if self.calibrator:
                if self.calibrator.state == CalibrationState.COLLECTING and self.calibrator._start_time:
                    dur = now - self.calibrator._start_time
                    calib_prog = min(1.0, max(0.0, dur / self.calibrator.config.target_duration_sec))
                elif self.calibrator.state == CalibrationState.COMPLETED:
                    calib_prog = 1.0

            # Preserve confirmed closed duration prior to tracking gap; unobserved gap time is excluded
            if self._eyes_closed_start_time is not None:
                if self._last_closed_seen_time is not None:
                    self._accumulated_closed_duration += max(0.0, self._last_closed_seen_time - self._eyes_closed_start_time)
                self._eyes_closed_start_time = None

            # Reset accumulated closure if gap exceeds grace period (microsecond precision tolerance)
            if self._last_closed_seen_time is not None:
                gap = max(0.0, now - self._last_closed_seen_time)
                if round(gap, 6) > self.config.tracking_gap_grace_sec:
                    self._accumulated_closed_duration = 0.0
                    self._last_closed_seen_time = None

            self._eyes_open_start_time = None

            # Record tracking loss in PERCLOS calculator (gap bounded)
            self.perclos_calculator.record_invalid_observation()
            perclos, p_valid, p_status, _, _ = self.perclos_calculator.compute_perclos(now)
            p_warning = p_valid and (perclos >= self.config.perclos_warning_threshold)

            return DetectionResult(
                timestamp=now,
                detection_state=self.detection_state,
                alarm_state=self.alarm_state,
                tracking_state=self.tracking_state,
                eye_state=self.eye_state,
                ear=0.0,
                left_ear=0.0,
                right_ear=0.0,
                blink_count=self.blink_count,
                closure_duration=0.0,
                tracking_lost_duration=lost_duration,
                session_id=self.session_id,
                landmarks=[],
                status_message=status_msg,
                perclos=perclos,
                perclos_valid=p_valid,
                perclos_status=p_status,
                perclos_window_duration=self.config.perclos_window_sec,
                perclos_warning=p_warning,
                baseline_active=(self.baseline is not None and self.baseline.is_valid),
                calibrated_ear_threshold=self.config.ear_threshold,
                calibration_state=calib_state,
                calibration_progress=calib_prog,
                calibration_failure_reason=calib_reason,
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                head_pose_valid=False,
                head_pose_status=PoseStatus.FACE_LOST.value,
                head_pose_reprojection_error=0.0,
                head_pose_confidence=0.0
            )

        # -------------------------------------------------------------
        # Branch 2: Face IS detected
        # -------------------------------------------------------------
        self._last_face_seen_time = now
        self._face_lost_start_time = None
        self.tracking_state = TrackingState.ACTIVE

        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye_points, self.config.min_eye_width)
        right_ear = calculate_ear(right_eye_points, self.config.min_eye_width)

        # If both geometries failed/degenerate
        if left_ear == 0.0 and right_ear == 0.0:
            avg_ear = 0.0
            self.eye_state = EyeState.UNKNOWN
            # Preserve confirmed closed duration prior to degenerate geometry; unobserved time excluded
            if self._eyes_closed_start_time is not None:
                if self._last_closed_seen_time is not None:
                    self._accumulated_closed_duration += max(0.0, self._last_closed_seen_time - self._eyes_closed_start_time)
                self._eyes_closed_start_time = None

            # Reset accumulated closure if gap exceeds grace period (microsecond precision tolerance)
            if self._last_closed_seen_time is not None:
                gap = max(0.0, now - self._last_closed_seen_time)
                if round(gap, 6) > self.config.tracking_gap_grace_sec:
                    self._accumulated_closed_duration = 0.0
                    self._last_closed_seen_time = None

            self._eyes_open_start_time = None
            current_closure_duration = 0.0
            status_msg = "Invalid eye landmark geometry."
            self.perclos_calculator.record_invalid_observation()

            if self.calibrator and self.calibrator.state == CalibrationState.COLLECTING:
                self.calibrator.add_sample(timestamp=now, ear=0.0, face_detected=True, is_valid_geometry=False)
                self._check_calibration_completion(now)
        else:
            # Average valid eye EARs
            if left_ear > 0.0 and right_ear > 0.0:
                avg_ear = (left_ear + right_ear) / 2.0
            else:
                avg_ear = max(left_ear, right_ear)

            # Record observation in PERCLOS calculator
            self.perclos_calculator.record_observation(ear=avg_ear, now=now)

            # Record observation in active baseline calibrator (using identical EAR)
            if self.calibrator and self.calibrator.state == CalibrationState.COLLECTING:
                is_geom_valid = (left_ear > 0.0 or right_ear > 0.0)
                self.calibrator.add_sample(
                    timestamp=now,
                    ear=avg_ear if is_geom_valid else 0.0,
                    face_detected=True,
                    is_valid_geometry=is_geom_valid
                )
                self._check_calibration_completion(now)

            # Determine instantaneous eye state
            is_closed = avg_ear < self.config.ear_threshold
            self.eye_state = EyeState.CLOSED if is_closed else EyeState.OPEN

            if is_closed:
                # Eye is currently closed
                self._eyes_open_start_time = None

                # If resuming after a tracking gap, check if gap exceeded grace period
                if self._last_closed_seen_time is not None and self._eyes_closed_start_time is None:
                    gap = max(0.0, now - self._last_closed_seen_time)
                    if round(gap, 6) > self.config.tracking_gap_grace_sec:
                        # Tracking gap exceeded grace period; discard prior accumulated closure
                        self._accumulated_closed_duration = 0.0
                        self._last_closed_seen_time = None

                if self._eyes_closed_start_time is None:
                    self._eyes_closed_start_time = now

                self._last_closed_seen_time = now
                current_closure_duration = self._accumulated_closed_duration + max(0.0, now - self._eyes_closed_start_time)

                # Check if closure duration exceeds threshold
                if current_closure_duration >= self.config.closure_duration_threshold:
                    self.detection_state = DetectionState.POSSIBLE_DROWSINESS
                    self.alarm_state = AlarmState.TRIGGERED
                    status_msg = (
                        f"Possible drowsiness detected! Eyes closed for {current_closure_duration:.1f}s."
                    )
                else:
                    if self.alarm_state == AlarmState.TRIGGERED:
                        self.detection_state = DetectionState.POSSIBLE_DROWSINESS
                    else:
                        self.detection_state = DetectionState.NORMAL
                    status_msg = f"Eyes closed ({current_closure_duration:.1f}s)..."
            else:
                # Eye is currently open
                current_closure_duration = 0.0
                if self._eyes_open_start_time is None:
                    self._eyes_open_start_time = now

                # If eyes were previously closed, check if it was a normal blink
                if self._eyes_closed_start_time is not None:
                    closed_duration = self._accumulated_closed_duration + max(0.0, now - self._eyes_closed_start_time)
                elif self._last_closed_seen_time is not None and round(max(0.0, now - self._last_closed_seen_time), 6) <= self.config.tracking_gap_grace_sec:
                    closed_duration = self._accumulated_closed_duration
                else:
                    closed_duration = 0.0

                if self.config.blink_min_duration <= closed_duration <= self.config.blink_max_duration:
                    self.blink_count += 1

                self._eyes_closed_start_time = None
                self._last_closed_seen_time = None
                self._accumulated_closed_duration = 0.0

                # Hysteresis for clearing an active alarm
                open_duration = max(0.0, now - self._eyes_open_start_time)
                if self.alarm_state == AlarmState.TRIGGERED:
                    if open_duration >= self.config.recovery_open_duration:
                        # Driver has maintained eyes open sufficiently to recover
                        self.alarm_state = AlarmState.OFF
                        self.detection_state = DetectionState.NORMAL
                        status_msg = "Alert cleared. Normal monitoring resumed."
                    else:
                        self.detection_state = DetectionState.POSSIBLE_DROWSINESS
                        status_msg = (
                            f"Eyes reopened. Verifying alertness ({open_duration:.1f}s / {self.config.recovery_open_duration:.1f}s)..."
                        )
                else:
                    self.detection_state = DetectionState.NORMAL
                    self.alarm_state = AlarmState.OFF
                    status_msg = "Monitoring normally."

        perclos, p_valid, p_status, _, _ = self.perclos_calculator.compute_perclos(now)
        p_warning = p_valid and (perclos >= self.config.perclos_warning_threshold)

        calib_state = self.calibrator.state.value if self.calibrator else "IDLE"
        calib_reason = self.calibrator.failure_reason.value if (self.calibrator and self.calibrator.state == CalibrationState.FAILED) else "NONE"
        calib_prog = 0.0
        if self.calibrator:
            if self.calibrator.state == CalibrationState.COLLECTING and self.calibrator._start_time:
                dur = now - self.calibrator._start_time
                calib_prog = min(1.0, max(0.0, dur / self.calibrator.config.target_duration_sec))
            elif self.calibrator.state == CalibrationState.COMPLETED:
                calib_prog = 1.0

        # Estimate 3D head pose as an independent observational signal
        if self.config.enable_head_pose and head_pose_points is not None:
            w_val = frame_width if (frame_width is not None and frame_width > 0) else 640
            h_val = frame_height if (frame_height is not None and frame_height > 0) else 480
            head_pose_res = self.head_pose_estimator.estimate_pose(
                head_pose_points,
                frame_width=w_val,
                frame_height=h_val
            )
        else:
            head_pose_res = HeadPoseResult(status=PoseStatus.UNAVAILABLE.value)

        return DetectionResult(
            timestamp=now,
            detection_state=self.detection_state,
            alarm_state=self.alarm_state,
            tracking_state=self.tracking_state,
            eye_state=self.eye_state,
            ear=avg_ear,
            left_ear=left_ear,
            right_ear=right_ear,
            blink_count=self.blink_count,
            closure_duration=current_closure_duration,
            tracking_lost_duration=0.0,
            session_id=self.session_id,
            landmarks=normalized_landmarks or [],
            status_message=status_msg,
            perclos=perclos,
            perclos_valid=p_valid,
            perclos_status=p_status,
            perclos_window_duration=self.config.perclos_window_sec,
            perclos_warning=p_warning,
            baseline_active=(self.baseline is not None and self.baseline.is_valid),
            calibrated_ear_threshold=self.config.ear_threshold,
            calibration_state=calib_state,
            calibration_progress=calib_prog,
            calibration_failure_reason=calib_reason,
            pitch=head_pose_res.pitch,
            yaw=head_pose_res.yaw,
            roll=head_pose_res.roll,
            head_pose_valid=head_pose_res.is_valid,
            head_pose_status=head_pose_res.status,
            head_pose_reprojection_error=head_pose_res.reprojection_error,
            head_pose_confidence=head_pose_res.confidence
        )
