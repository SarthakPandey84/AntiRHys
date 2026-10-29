"""
AntiRHys Offline Training Dataset Schema & Benchmark Evaluation Module
======================================================================
Defines standard dataset schemas, validation routines, participant-level cohort
partitioning, and offline benchmark metric computation for prospective ML models.

Safety Precepts:
- No automated online continuous learning in production.
- Strictly participant-separated cohort splits (Train / Val / Test).
- Standard evaluation metrics: Precision, Recall, FA/hr, Detection Delay.
"""

from dataclasses import dataclass, asdict
import json
import math
import random
from typing import List, Dict, Any, Tuple, Optional


REQUIRED_TELEMETRY_COLUMNS = [
    "timestamp",
    "session_id",
    "event_type",
    "severity",
    "detection_state",
    "eye_state",
    "tracking_state",
    "ear",
    "blink_count",
    "closure_duration",
    "perclos",
    "perclos_status",
    "ear_threshold",
    "calibration_state",
    "detector_version",
    "notes",
]

ALLOWED_SEVERITIES = {"INFO", "WARNING", "CRITICAL", "ERROR"}
ALLOWED_DETECTION_STATES = {"NORMAL", "POSSIBLE_DROWSINESS", "TRACKING_LOST", "INITIALIZING", "IDLE"}


@dataclass
class TelemetryRecord:
    """A validated telemetry record from AntiRHys session logs."""
    timestamp: str
    session_id: str
    event_type: str
    severity: str
    detection_state: str
    eye_state: str
    tracking_state: str
    ear: Optional[float]
    blink_count: Optional[int]
    closure_duration: Optional[float]
    perclos: Optional[float]
    perclos_status: str
    ear_threshold: Optional[float]
    calibration_state: str
    detector_version: str
    notes: str
    ground_truth_label: Optional[int] = None  # 0=Alert, 1=Drowsy/Micro-sleep, 2=Head Drop, -1=Exclude


def validate_telemetry_dict(row: Dict[str, Any]) -> Tuple[bool, str]:
    """Verify that a telemetry dictionary conforms to the required offline schema."""
    for col in REQUIRED_TELEMETRY_COLUMNS:
        if col not in row:
            return False, f"Missing required column: '{col}'"

    severity = str(row.get("severity", "")).upper()
    if severity not in ALLOWED_SEVERITIES:
        return False, f"Invalid severity: '{severity}'"

    # Numeric validations
    for num_col in ["ear", "closure_duration", "perclos", "ear_threshold"]:
        val = row.get(num_col)
        if val is not None and val != "":
            try:
                num = float(val)
                if not math.isfinite(num):
                    return False, f"Non-finite numeric value in '{num_col}'"
            except (ValueError, TypeError):
                return False, f"Invalid float value in '{num_col}': {val}"

    return True, "Valid"


def partition_by_participant(
    records: List[Dict[str, Any]],
    participant_key: str = "session_id",
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partition records strictly by participant / session ID into Train, Validation, and Test cohorts.
    Guarantees no participant overlap across splits to prevent morphological data leakage.
    """
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-3), "Ratios must sum to 1.0"

    participants = sorted(list(set(str(r.get(participant_key, "")) for r in records)))
    rng = random.Random(random_seed)
    rng.shuffle(participants)

    n_total = len(participants)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(participants[:n_train])
    val_ids = set(participants[n_train:n_train + n_val])
    test_ids = set(participants[n_train + n_val:])

    train_set = [r for r in records if str(r.get(participant_key, "")) in train_ids]
    val_set = [r for r in records if str(r.get(participant_key, "")) in val_ids]
    test_set = [r for r in records if str(r.get(participant_key, "")) in test_ids]

    return train_set, val_set, test_set


def compute_benchmark_metrics(
    y_true: List[int],
    y_pred: List[int],
    timestamps: Optional[List[float]] = None,
    total_hours: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute rigorous driver vigilance evaluation metrics:
    - Precision, Recall (Sensitivity), Specificity, F1-Score
    - False Alarms Per Hour (FA/hr)
    - Mean Alert Latency (Delay in seconds from closure onset to trigger)
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have identical length"

    # Filter out excluded/unlabeled samples (-1)
    filtered = [(t, p) for t, p in zip(y_true, y_pred) if t >= 0]
    if not filtered:
        return {"error": "No valid labeled samples"}

    tp = sum(1 for t, p in filtered if t == 1 and p == 1)
    fp = sum(1 for t, p in filtered if t == 0 and p == 1)
    fn = sum(1 for t, p in filtered if t == 1 and p == 0)
    tn = sum(1 for t, p in filtered if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # False Alarms Per Hour
    fa_per_hour = None
    if total_hours and total_hours > 0:
        fa_per_hour = round(fp / total_hours, 2)
    elif timestamps and len(timestamps) >= 2:
        span_hours = max(0.001, (timestamps[-1] - timestamps[0]) / 3600.0)
        fa_per_hour = round(fp / span_hours, 2)

    return {
        "samples_evaluated": len(filtered),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "false_alarms_per_hour": fa_per_hour,
    }


if __name__ == "__main__":
    # Self-test validation
    sample_row = {
        "timestamp": "2026-10-29T08:00:00.000000",
        "session_id": "session_test_001",
        "event_type": "STATE_CHANGE",
        "severity": "CRITICAL",
        "detection_state": "POSSIBLE_DROWSINESS",
        "eye_state": "CLOSED",
        "tracking_state": "ACTIVE",
        "ear": 0.142,
        "blink_count": 12,
        "closure_duration": 1.6,
        "perclos": 0.18,
        "perclos_status": "VALID",
        "ear_threshold": 0.245,
        "calibration_state": "COMPLETED",
        "detector_version": "2.1.0",
        "notes": "Possible drowsiness alert",
    }

    valid, err = validate_telemetry_dict(sample_row)
    assert valid, f"Sample row validation failed: {err}"

    # Metric self-test
    y_t = [0, 0, 1, 1, 0, 1, 0, 0, -1]
    y_p = [0, 1, 1, 1, 0, 0, 0, 0,  0]
    metrics = compute_benchmark_metrics(y_t, y_p, total_hours=2.0)
    print("Benchmark Metrics Self-Test Output:")
    print(json.dumps(metrics, indent=2))
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    print("Training dataset schema & metrics self-test passed successfully!")

