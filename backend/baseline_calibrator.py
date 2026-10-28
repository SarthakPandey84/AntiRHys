"""
AntiRHys Personalized Driver Baseline Calibrator
================================================
Provides a privacy-preserving, transparent, and bounded calibration engine
for learning driver-specific ocular baselines (e.g. open-eye EAR distribution)
under controlled, consenting conditions.

Core Scientific & Ethical Guardrails:
1. Individual Variation over Demographic Proxies:
   - Does NOT use race, ethnicity, age, or gender as proxies for eye geometry or drowsiness.
   - Measures individual physiological features (EAR distribution, blink timing) directly.

2. Quality & Drowsiness Gating:
   - Requires minimum valid observation time and >= 80% face tracking stability.
   - Rejects calibration sessions exhibiting signs of existing drowsiness (e.g. prolonged
     eye closures or excessive closure fractions).

3. Bounded Adaptive Thresholds:
   - Adapts eye-closure threshold relative to individual open-eye baseline (e.g. 70% of median).
   - Strictly clamped to a safe engineering envelope [MIN_THRESHOLD=0.18, MAX_THRESHOLD=0.32]
     to prevent normalizing dangerous drowsiness or triggering constant false alarms.

4. Privacy-by-Design:
   - Retains NO facial imagery, video frames, or raw landmark coordinate arrays.
   - Stores only minimal anonymous scalar summary statistics (mean, std, median, percentiles).
   - Provides an immediate reset/delete interface.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import math
import statistics
import time
from typing import List, Optional, Dict, Any


class CalibrationState(str, Enum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CalibrationFailureReason(str, Enum):
    NONE = "NONE"
    INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"
    INSUFFICIENT_DURATION = "INSUFFICIENT_DURATION"
    LOW_TRACKING_RATIO = "LOW_TRACKING_RATIO"
    SUSPECTED_DROWSINESS = "SUSPECTED_DROWSINESS"
    EXTREME_EAR_OUTLIER = "EXTREME_EAR_OUTLIER"
    USER_ABORTED = "USER_ABORTED"


@dataclass
class CalibrationConfig:
    """Configuration constraints for individual baseline calibration."""
    # Target duration in seconds of active attentive calibration
    target_duration_sec: float = 12.0
    # Minimum observation duration required to produce a valid baseline
    min_duration_sec: float = 8.0
    # Minimum valid observation samples
    min_valid_samples: int = 50
    # Minimum ratio of successfully tracked frames during calibration window
    min_tracking_ratio: float = 0.80
    # Proportional multiplier for closure threshold (e.g., 70% of baseline open EAR)
    closure_multiplier: float = 0.70
    # Absolute minimum floor for personalized closure threshold
    min_closure_threshold: float = 0.18
    # Absolute maximum ceiling for personalized closure threshold
    max_closure_threshold: float = 0.32
    # Default uncalibrated fallback threshold
    fallback_closure_threshold: float = 0.25
    # Maximum tolerable closure duration (sec) during calibration before flagging drowsiness
    max_closure_event_sec: float = 0.75
    # Maximum fraction of closed frames during calibration before aborting
    max_closed_frame_fraction: float = 0.15
    # Minimum plausible mean open-eye EAR to prevent calibrating against closed eyes
    min_plausible_open_ear: float = 0.20
    # Maximum plausible mean open-eye EAR (rejects severe optical distortion)
    max_plausible_open_ear: float = 0.50


@dataclass
class DriverBaseline:
    """
    Privacy-preserving summary of a driver's individual ocular baseline.
    Contains strictly scalar distribution parameters and no raw facial or video data.
    """
    driver_id: str
    calibrated_at: float               # Unix timestamp of calibration
    is_valid: bool                     # True if calibration passed all validation gates
    failure_reason: CalibrationFailureReason
    sample_count: int
    duration_sec: float
    tracking_ratio: float
    mean_open_ear: float
    std_open_ear: float
    median_open_ear: float
    p10_open_ear: float
    p90_open_ear: float
    personalized_closure_threshold: float
    personalized_perclos_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize baseline parameters to dictionary for transparent storage."""
        data = asdict(self)
        data["failure_reason"] = self.failure_reason.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriverBaseline":
        """Deserialize baseline parameters from dictionary."""
        d = dict(data)
        d["failure_reason"] = CalibrationFailureReason(d.get("failure_reason", "NONE"))
        return cls(**d)

    @classmethod
    def default_fallback(cls, driver_id: str = "anonymous", reason: CalibrationFailureReason = CalibrationFailureReason.NONE) -> "DriverBaseline":
        """Generate safe default fallback when calibration is not performed or fails."""
        return cls(
            driver_id=driver_id,
            calibrated_at=time.time(),
            is_valid=False,
            failure_reason=reason,
            sample_count=0,
            duration_sec=0.0,
            tracking_ratio=0.0,
            mean_open_ear=0.30,
            std_open_ear=0.03,
            median_open_ear=0.30,
            p10_open_ear=0.26,
            p90_open_ear=0.34,
            personalized_closure_threshold=0.25,
            personalized_perclos_threshold=0.20,
        )


@dataclass
class _RawSample:
    timestamp: float
    ear: float
    face_detected: bool
    is_valid_geometry: bool


class BaselineCalibrator:
    """
    Orchestrates the active calibration process.
    Collects raw observations over a controlled window, evaluates quality criteria,
    verifies lack of drowsiness, and calculates individual statistical thresholds.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.state: CalibrationState = CalibrationState.IDLE
        self.failure_reason: CalibrationFailureReason = CalibrationFailureReason.NONE
        self._samples: List[_RawSample] = []
        self._start_time: Optional[float] = None
        self._driver_id: str = "anonymous"

    def start_calibration(self, driver_id: str = "anonymous", start_timestamp: Optional[float] = None) -> None:
        """Start or restart a baseline calibration window."""
        self.state = CalibrationState.COLLECTING
        self.failure_reason = CalibrationFailureReason.NONE
        self._samples.clear()
        self._driver_id = driver_id or "anonymous"
        self._start_time = start_timestamp

    def add_sample(
        self,
        timestamp: float,
        ear: float,
        face_detected: bool,
        is_valid_geometry: bool = True
    ) -> None:
        """
        Record a sample during calibration.

        Args:
            timestamp: Monotonic timestamp of the frame.
            ear: Computed Eye Aspect Ratio (0.0 if face missing).
            face_detected: True if face was tracked.
            is_valid_geometry: True if landmark geometry passed sanity checks.
        """
        if self.state != CalibrationState.COLLECTING:
            return

        # Lazily initialize start_time on first received sample if not explicitly given
        if self._start_time is None and math.isfinite(timestamp):
            self._start_time = timestamp

        # Guard against non-finite values (NaN, Inf) or negative inputs
        if not (math.isfinite(timestamp) and math.isfinite(ear)) or timestamp < 0.0 or ear < 0.0:
            self._samples.append(_RawSample(
                timestamp=timestamp if math.isfinite(timestamp) else 0.0,
                ear=0.0,
                face_detected=False,
                is_valid_geometry=False
            ))
            return

        self._samples.append(_RawSample(
            timestamp=timestamp,
            ear=ear,
            face_detected=face_detected,
            is_valid_geometry=is_valid_geometry
        ))

    def abort(self) -> None:
        """User or system abort of the current calibration."""
        self.state = CalibrationState.FAILED
        self.failure_reason = CalibrationFailureReason.USER_ABORTED

    def reset(self) -> None:
        """Clear all collected samples and reset to IDLE."""
        self.state = CalibrationState.IDLE
        self.failure_reason = CalibrationFailureReason.NONE
        self._samples.clear()
        self._start_time = None

    def finalize(self, end_timestamp: Optional[float] = None) -> DriverBaseline:
        """
        Evaluate collected calibration data and generate a DriverBaseline.
        If any validation gate fails, returns a baseline with is_valid=False and fallback values.
        """
        if self.state == CalibrationState.FAILED:
            return DriverBaseline.default_fallback(self._driver_id, self.failure_reason)

        now = end_timestamp if end_timestamp is not None else time.monotonic()
        total_duration = (now - self._start_time) if self._start_time is not None else 0.0

        total_frames = len(self._samples)
        if total_frames < self.config.min_valid_samples:
            self.state = CalibrationState.FAILED
            self.failure_reason = CalibrationFailureReason.INSUFFICIENT_SAMPLES
            return DriverBaseline.default_fallback(self._driver_id, self.failure_reason)

        if total_duration < self.config.min_duration_sec:
            self.state = CalibrationState.FAILED
            self.failure_reason = CalibrationFailureReason.INSUFFICIENT_DURATION
            return DriverBaseline.default_fallback(self._driver_id, self.failure_reason)

        # 1. Evaluate tracking ratio
        tracked_samples = [s for s in self._samples if s.face_detected and s.is_valid_geometry]
        tracking_ratio = len(tracked_samples) / float(total_frames) if total_frames > 0 else 0.0

        if tracking_ratio < self.config.min_tracking_ratio:
            self.state = CalibrationState.FAILED
            self.failure_reason = CalibrationFailureReason.LOW_TRACKING_RATIO
            return DriverBaseline.default_fallback(self._driver_id, self.failure_reason)

        # 2. Extract valid EAR values
        ear_values = [s.ear for s in tracked_samples if s.ear > 0.0]
        if len(ear_values) < self.config.min_valid_samples:
            self.state = CalibrationState.FAILED
            self.failure_reason = CalibrationFailureReason.INSUFFICIENT_SAMPLES
            return DriverBaseline.default_fallback(self._driver_id, self.failure_reason)

        # Sort EAR values for percentiles
        sorted_ears = sorted(ear_values)
        n = len(sorted_ears)
        median_ear = statistics.median(sorted_ears)
        mean_ear = statistics.mean(sorted_ears)
        std_ear = statistics.stdev(sorted_ears) if n > 1 else 0.0

        p10_idx = max(0, int(0.10 * n))
        p90_idx = min(n - 1, int(0.90 * n))
        p10_ear = sorted_ears[p10_idx]
        p90_ear = sorted_ears[p90_idx]

        # 3. Check for plausible physiological open-eye EAR
        if median_ear < self.config.min_plausible_open_ear or median_ear > self.config.max_plausible_open_ear:
            self.state = CalibrationState.FAILED
            self.failure_reason = CalibrationFailureReason.EXTREME_EAR_OUTLIER
            return DriverBaseline.default_fallback(self._driver_id, self.failure_reason)

        # 4. Check for suspected drowsiness during calibration:
        # A subject must be awake and attentive during calibration.
        # Check: Fraction of closed frames (EAR < provisional threshold) and longest consecutive closure.
        provisional_thresh = median_ear * self.config.closure_multiplier
        closed_count = sum(1 for e in ear_values if e < provisional_thresh)
        closed_fraction = closed_count / float(len(ear_values))

        # Check consecutive closure duration
        max_consecutive_closure_sec = 0.0
        current_closure_start = None
        for s in tracked_samples:
            if s.ear < provisional_thresh:
                if current_closure_start is None:
                    current_closure_start = s.timestamp
                else:
                    dur = s.timestamp - current_closure_start
                    if dur > max_consecutive_closure_sec:
                        max_consecutive_closure_sec = dur
            else:
                current_closure_start = None

        if (
            closed_fraction > self.config.max_closed_frame_fraction
            or max_consecutive_closure_sec > self.config.max_closure_event_sec
        ):
            self.state = CalibrationState.FAILED
            self.failure_reason = CalibrationFailureReason.SUSPECTED_DROWSINESS
            return DriverBaseline.default_fallback(self._driver_id, self.failure_reason)

        # 5. Compute personalized threshold with strict clamping
        raw_closure_thresh = median_ear * self.config.closure_multiplier
        clamped_closure_thresh = max(
            self.config.min_closure_threshold,
            min(self.config.max_closure_threshold, raw_closure_thresh)
        )

        self.state = CalibrationState.COMPLETED
        self.failure_reason = CalibrationFailureReason.NONE

        return DriverBaseline(
            driver_id=self._driver_id,
            calibrated_at=time.time(),
            is_valid=True,
            failure_reason=CalibrationFailureReason.NONE,
            sample_count=len(tracked_samples),
            duration_sec=round(total_duration, 2),
            tracking_ratio=round(tracking_ratio, 3),
            mean_open_ear=round(mean_ear, 4),
            std_open_ear=round(std_ear, 4),
            median_open_ear=round(median_ear, 4),
            p10_open_ear=round(p10_ear, 4),
            p90_open_ear=round(p90_ear, 4),
            personalized_closure_threshold=round(clamped_closure_thresh, 4),
            personalized_perclos_threshold=round(self.config.fallback_closure_threshold * 0.8, 4),
        )


def save_baseline_to_file(baseline: DriverBaseline, filepath: str) -> bool:
    """
    Safely persist scalar baseline statistics to a JSON file.
    Privacy-by-design: persists only scalar distribution parameters, no image or facial data.
    """
    import json
    import os
    try:
        dirname = os.path.dirname(os.path.abspath(filepath))
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(baseline.to_dict(), f, indent=2)
        return True
    except Exception as e:
        print(f"[BaselineCalibrator] Failed to save baseline to {filepath}: {e}")
        return False


def load_baseline_from_file(filepath: str) -> Optional[DriverBaseline]:
    """
    Safely load a DriverBaseline from a JSON file.
    Returns None on missing files, corrupt JSON, missing keys, or non-finite numbers.
    """
    import json
    import os
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None

        # Validate presence and finiteness of numerical fields
        numeric_fields = [
            "mean_open_ear", "std_open_ear", "median_open_ear",
            "p10_open_ear", "p90_open_ear",
            "personalized_closure_threshold", "personalized_perclos_threshold"
        ]
        for field in numeric_fields:
            val = data.get(field)
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                return None

        return DriverBaseline.from_dict(data)
    except Exception as e:
        print(f"[BaselineCalibrator] Failed to load baseline from {filepath}: {e}")
        return None
