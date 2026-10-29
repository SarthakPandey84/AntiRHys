"""
AntiRHys Personalization Profile Manager
========================================
Provides versioned, quality-gated persistence for individual driver ocular baselines.

Scientific & Privacy Guardrails:
1. Versioned Profile with Minimal Necessary Data:
   - Stores only scalar summary distribution parameters (open EAR, closure threshold).
   - Never stores facial imagery, video frames, or raw landmark coordinate streams.
2. Conservative Bounded Updates:
   - Closure threshold is strictly clamped to [0.180, 0.320].
   - Rejects updates exhibiting tracking loss (< 80%), eye closure during calibration,
     insufficient samples (< 50), or implausible EAR (< 0.20 or > 0.50).
3. Last Known-Good Preservation:
   - If a calibration attempt is rejected, the existing valid profile is preserved untouched.
4. User-Controlled Reset & Recalibration:
   - Immediate reset endpoint restores uncalibrated factory default (0.250).
"""

from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Dict, Any, Optional, Tuple

from baseline_calibrator import DriverBaseline, CalibrationFailureReason

DEFAULT_DETECTOR_VERSION = "2.1.0"
MIN_THRESHOLD_BOUND = 0.180
MAX_THRESHOLD_BOUND = 0.320
FACTORY_DEFAULT_THRESHOLD = 0.250


@dataclass
class PersonalizationProfile:
    """Versioned driver profile containing individual baseline parameters."""
    profile_version: int = 1
    driver_id: str = "driver"
    is_active: bool = False
    baseline_open_ear: Optional[float] = None
    personalized_threshold: float = FACTORY_DEFAULT_THRESHOLD
    min_threshold_bound: float = MIN_THRESHOLD_BOUND
    max_threshold_bound: float = MAX_THRESHOLD_BOUND
    sample_count: int = 0
    duration_sec: float = 0.0
    tracking_ratio: float = 0.0
    calibration_timestamp: Optional[float] = None
    quality_status: str = "UNAVAILABLE"  # HIGH | ACCEPTABLE | UNAVAILABLE
    detector_version: str = DEFAULT_DETECTOR_VERSION
    update_reason: str = "Default factory threshold (0.250)"
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def default_factory(cls, driver_id: str = "driver") -> "PersonalizationProfile":
        now = time.time()
        return cls(
            profile_version=1,
            driver_id=driver_id,
            is_active=False,
            baseline_open_ear=None,
            personalized_threshold=FACTORY_DEFAULT_THRESHOLD,
            min_threshold_bound=MIN_THRESHOLD_BOUND,
            max_threshold_bound=MAX_THRESHOLD_BOUND,
            sample_count=0,
            duration_sec=0.0,
            tracking_ratio=0.0,
            calibration_timestamp=None,
            quality_status="UNAVAILABLE",
            detector_version=DEFAULT_DETECTOR_VERSION,
            update_reason="Default factory threshold (0.250)",
            created_at=now,
            updated_at=now,
        )


class ProfileManager:
    """Thread-safe manager for versioned driver personalization profiles."""

    def __init__(self, profile_filepath: str = "driver_profile.json"):
        self.filepath = Path(profile_filepath)
        self._lock = threading.Lock()
        self._current_profile: PersonalizationProfile = self._load_or_default()

    def _load_or_default(self) -> PersonalizationProfile:
        """Load persisted profile from disk or initialize safe default."""
        if not self.filepath.exists():
            return PersonalizationProfile.default_factory()

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return PersonalizationProfile.default_factory()

            # Validate numerical types and bounds
            thresh = data.get("personalized_threshold", FACTORY_DEFAULT_THRESHOLD)
            if not isinstance(thresh, (int, float)) or not math.isfinite(thresh):
                thresh = FACTORY_DEFAULT_THRESHOLD
            thresh = max(MIN_THRESHOLD_BOUND, min(MAX_THRESHOLD_BOUND, float(thresh)))

            open_ear = data.get("baseline_open_ear")
            if open_ear is not None:
                if not isinstance(open_ear, (int, float)) or not math.isfinite(open_ear) or open_ear <= 0:
                    open_ear = None
                else:
                    open_ear = float(open_ear)

            return PersonalizationProfile(
                profile_version=int(data.get("profile_version", 1)),
                driver_id=str(data.get("driver_id", "driver"))[:64],
                is_active=bool(data.get("is_active", False)),
                baseline_open_ear=open_ear,
                personalized_threshold=thresh,
                min_threshold_bound=MIN_THRESHOLD_BOUND,
                max_threshold_bound=MAX_THRESHOLD_BOUND,
                sample_count=int(data.get("sample_count", 0)),
                duration_sec=float(data.get("duration_sec", 0.0)),
                tracking_ratio=float(data.get("tracking_ratio", 0.0)),
                calibration_timestamp=data.get("calibration_timestamp"),
                quality_status=str(data.get("quality_status", "UNAVAILABLE")),
                detector_version=str(data.get("detector_version", DEFAULT_DETECTOR_VERSION)),
                update_reason=str(data.get("update_reason", "Loaded from storage")),
                created_at=float(data.get("created_at", time.time())),
                updated_at=float(data.get("updated_at", time.time())),
            )
        except Exception as e:
            print(f"[ProfileManager] Warning: failed to parse profile file, using default: {e}")
            return PersonalizationProfile.default_factory()

    def _save_to_disk(self) -> bool:
        """Persist current profile atomically to disk."""
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.filepath.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._current_profile.to_dict(), f, indent=2)
            # Atomic replace
            os.replace(tmp_path, self.filepath)
            return True
        except Exception as e:
            print(f"[ProfileManager] Error saving profile to {self.filepath}: {e}")
            return False

    def get_profile(self) -> Dict[str, Any]:
        """Return the current versioned profile dictionary."""
        with self._lock:
            return self._current_profile.to_dict()

    def update_from_baseline(
        self,
        baseline: DriverBaseline,
        reason: str = "Attentive baseline calibration"
    ) -> Tuple[bool, str]:
        """
        Evaluate and apply a DriverBaseline update to the profile.
        Strictly rejects invalid, drowsy, or out-of-envelope baselines,
        preserving the existing profile untouched if rejected.
        """
        with self._lock:
            # Gate 1: Baseline validity
            if not baseline or not baseline.is_valid:
                reason_str = baseline.failure_reason.value if baseline else "NULL_BASELINE"
                return False, f"Rejected: baseline validity check failed ({reason_str})."

            # Gate 2: Sufficient valid observations
            if baseline.sample_count < 50:
                return False, f"Rejected: insufficient calibration samples ({baseline.sample_count} < 50)."

            # Gate 3: Face tracking stability ratio
            if baseline.tracking_ratio < 0.80:
                return False, f"Rejected: tracking stability below 80% ({baseline.tracking_ratio * 100:.1f}%)."

            # Gate 4: Plausible physiological limits for open-eye EAR
            if not (0.20 <= baseline.mean_open_ear <= 0.50):
                return False, f"Rejected: open-eye EAR ({baseline.mean_open_ear:.3f}) outside plausible limits [0.20, 0.50]."

            # Gate 5: Bounded closure threshold clamp
            clamped_thresh = max(MIN_THRESHOLD_BOUND, min(MAX_THRESHOLD_BOUND, float(baseline.personalized_closure_threshold)))

            quality_label = "HIGH" if (baseline.tracking_ratio >= 0.95 and baseline.sample_count >= 80) else "ACCEPTABLE"
            now = time.time()
            new_version = self._current_profile.profile_version + 1

            # Atomically update profile
            self._current_profile = PersonalizationProfile(
                profile_version=new_version,
                driver_id=baseline.driver_id or "driver",
                is_active=True,
                baseline_open_ear=round(baseline.mean_open_ear, 3),
                personalized_threshold=round(clamped_thresh, 3),
                min_threshold_bound=MIN_THRESHOLD_BOUND,
                max_threshold_bound=MAX_THRESHOLD_BOUND,
                sample_count=baseline.sample_count,
                duration_sec=round(baseline.duration_sec, 1),
                tracking_ratio=round(baseline.tracking_ratio, 3),
                calibration_timestamp=baseline.calibrated_at or now,
                quality_status=quality_label,
                detector_version=DEFAULT_DETECTOR_VERSION,
                update_reason=reason,
                created_at=self._current_profile.created_at or now,
                updated_at=now,
            )

            self._save_to_disk()
            return True, "Personalization profile successfully updated."

    def reset_profile(self, driver_id: str = "driver", reason: str = "Reset to factory defaults") -> Dict[str, Any]:
        """Reset profile to uncalibrated factory defaults (0.250)."""
        with self._lock:
            now = time.time()
            new_version = self._current_profile.profile_version + 1
            self._current_profile = PersonalizationProfile(
                profile_version=new_version,
                driver_id=driver_id,
                is_active=False,
                baseline_open_ear=None,
                personalized_threshold=FACTORY_DEFAULT_THRESHOLD,
                min_threshold_bound=MIN_THRESHOLD_BOUND,
                max_threshold_bound=MAX_THRESHOLD_BOUND,
                sample_count=0,
                duration_sec=0.0,
                tracking_ratio=0.0,
                calibration_timestamp=None,
                quality_status="UNAVAILABLE",
                detector_version=DEFAULT_DETECTOR_VERSION,
                update_reason=reason,
                created_at=self._current_profile.created_at or now,
                updated_at=now,
            )
            self._save_to_disk()
            return self._current_profile.to_dict()

    def delete_profile(self) -> bool:
        """Completely remove profile file and reset in-memory state."""
        with self._lock:
            self._current_profile = PersonalizationProfile.default_factory()
            try:
                if self.filepath.exists():
                    self.filepath.unlink()
                return True
            except Exception as e:
                print(f"[ProfileManager] Error deleting profile file: {e}")
                return False

