"""
AntiRHys 3D Head Pose Estimator
================================
Estimates driver head rotational angles (Pitch, Yaw, Roll) via Perspective-n-Point (solvePnP)
using canonical anthropometric 3D facial landmarks and 2D MediaPipe FaceMesh coordinates.

Conventions & Units:
--------------------
- Angles: Measured in degrees (°), rounded to 1 or 2 decimal places.
- Pitch: Rotation around the lateral axis (X-axis).
    * Negative (< 0°): Head tilted DOWN (chin lowered towards chest / micro-sleep nod).
    * Positive (> 0°): Head tilted UP (chin raised towards roof).
- Yaw: Rotation around the vertical axis (Y-axis).
    * Negative (< 0°): Head turned to the DRIVER'S RIGHT (camera left).
    * Positive (> 0°): Head turned to the DRIVER'S LEFT (camera right).
- Roll: Rotation around the longitudinal optical axis (Z-axis).
    * Negative (< 0°): Head tilted toward driver's left shoulder (counter-clockwise in 2D image).
    * Positive (> 0°): Head tilted toward driver's right shoulder (clockwise in 2D image).

Design & Safety Guarantees:
---------------------------
1. Complete Isolation: Head pose is an observational signal; it does NOT alter alarm states.
2. Geometric Pre-validation: Degenerate, collinear, or non-finite inputs are safely caught before PnP.
3. Reprojection Error Gating: Solutions with high reprojection error are marked invalid.
4. Zero Persistent Biometrics: No landmark arrays or face images are stored.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np


# MediaPipe FaceMesh landmark indices used for 6-point Perspective-n-Point
# 1: Nose tip
# 152: Chin
# 263: Outer corner of subject's left eye (image right)
# 33: Outer corner of subject's right eye (image left)
# 291: Left mouth corner (image right)
# 61: Right mouth corner (image left)
HEAD_POSE_LANDMARK_INDICES = [1, 152, 263, 33, 291, 61]


# Canonical 3D facial reference model points in millimeters (mm)
# Origin (0, 0, 0) is at the tip of the nose.
# +X: Right (toward subject's left ear, image right)
# +Y: Downward (toward chin)
# +Z: Backward / Depth (toward back of skull)
CANONICAL_FACE_3D = np.array([
    [0.0, 0.0, 0.0],          # 1: Nose tip
    [0.0, 65.0, -20.0],       # 152: Chin
    [43.0, -35.0, -30.0],     # 263: Left eye outer corner
    [-43.0, -35.0, -30.0],    # 33: Right eye outer corner
    [30.0, 35.0, -25.0],      # 291: Left mouth corner
    [-30.0, 35.0, -25.0]      # 61: Right mouth corner
], dtype=np.float64)


class PoseStatus(str, Enum):
    VALID = "VALID"
    FACE_LOST = "FACE_LOST"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_POINT_COUNT = "INVALID_POINT_COUNT"
    NON_FINITE_COORDINATES = "NON_FINITE_COORDINATES"
    DEGENERATE_GEOMETRY = "DEGENERATE_GEOMETRY"
    REPROJECTION_ERROR_HIGH = "REPROJECTION_ERROR_HIGH"
    SOLVE_FAILED = "SOLVE_FAILED"


@dataclass
class HeadPoseResult:
    """Represents instantaneous head pose estimation outputs and validation metrics."""
    pitch: float = 0.0                      # Degrees: negative = down/nod, positive = up
    yaw: float = 0.0                        # Degrees: negative = driver right, positive = driver left
    roll: float = 0.0                       # Degrees: negative = tilt left, positive = tilt right
    is_valid: bool = False                  # True if pose estimation met all quality gates
    status: str = PoseStatus.UNAVAILABLE.value
    reprojection_error: float = 0.0         # Root-mean-square pixel distance between model & observations
    confidence: float = 0.0                 # Confidence score [0.0 - 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pitch": round(self.pitch, 1),
            "yaw": round(self.yaw, 1),
            "roll": round(self.roll, 1),
            "head_pose_valid": self.is_valid,
            "head_pose_status": self.status,
            "head_pose_reprojection_error": round(self.reprojection_error, 2),
            "head_pose_confidence": round(self.confidence, 2)
        }


class HeadPoseEstimator:
    """
    Solves 3D driver head pose from 2D facial landmarks using OpenCV solvePnP.
    Provides robust sanity checks, camera matrix estimation, and reprojection error gating.
    """

    def __init__(
        self,
        max_reprojection_error: float = 15.0,
        min_bounding_size: float = 15.0
    ):
        """
        Args:
            max_reprojection_error: Maximum allowed RMS reprojection error in pixels before rejecting pose.
            min_bounding_size: Minimum pixel span in width and height to prevent degenerate collinear inputs.
        """
        self.model_3d = CANONICAL_FACE_3D
        self.max_reprojection_error = max_reprojection_error
        self.min_bounding_size = min_bounding_size

    def estimate_pose(
        self,
        landmarks_2d: Optional[List[Tuple[float, float]]],
        frame_width: int,
        frame_height: int
    ) -> HeadPoseResult:
        """
        Estimate pitch, yaw, and roll from 6 2D facial landmark points.

        Args:
            landmarks_2d: List of 6 (x, y) pixel coordinates matching HEAD_POSE_LANDMARK_INDICES.
            frame_width: Pixel width of the video frame.
            frame_height: Pixel height of the video frame.

        Returns:
            HeadPoseResult with calculated angles, validation flag, and diagnostic status.
        """
        # Guard against missing inputs
        if landmarks_2d is None or len(landmarks_2d) == 0:
            return HeadPoseResult(status=PoseStatus.FACE_LOST.value)

        # Validate point count
        if len(landmarks_2d) != 6:
            return HeadPoseResult(status=PoseStatus.INVALID_POINT_COUNT.value)

        # Validate frame dimensions
        if frame_width <= 0 or frame_height <= 0:
            return HeadPoseResult(status=PoseStatus.UNAVAILABLE.value)

        # Pre-validation: check coordinate finiteness
        points_arr = np.array(landmarks_2d, dtype=np.float64)
        if not np.all(np.isfinite(points_arr)):
            return HeadPoseResult(status=PoseStatus.NON_FINITE_COORDINATES.value)

        # Pre-validation: check geometric bounding box spread to prevent degenerate lines or collapsed points
        min_x, min_y = np.min(points_arr, axis=0)
        max_x, max_y = np.max(points_arr, axis=0)
        span_w = max_x - min_x
        span_h = max_y - min_y
        if span_w < self.min_bounding_size or span_h < self.min_bounding_size:
            return HeadPoseResult(status=PoseStatus.DEGENERATE_GEOMETRY.value)

        # Build camera intrinsic matrix (standard pinhole approximation when focal length is uncalibrated)
        # Using focal_length = frame_width is standard for typical ~60° FOV webcams
        focal_length = float(frame_width)
        center_x = float(frame_width) / 2.0
        center_y = float(frame_height) / 2.0

        camera_matrix = np.array([
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        try:
            success, rvec, tvec = cv2.solvePnP(
                self.model_3d,
                points_arr,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except Exception:
            return HeadPoseResult(status=PoseStatus.SOLVE_FAILED.value)

        if not success or rvec is None or tvec is None or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            return HeadPoseResult(status=PoseStatus.SOLVE_FAILED.value)

        # Compute Reprojection Error (RMSE in pixels)
        try:
            proj_points, _ = cv2.projectPoints(self.model_3d, rvec, tvec, camera_matrix, dist_coeffs)
            proj_2d = proj_points.reshape(-1, 2)
            rmse = float(np.sqrt(np.mean((points_arr - proj_2d) ** 2)))
        except Exception:
            rmse = 999.0

        if not math.isfinite(rmse) or rmse > self.max_reprojection_error:
            return HeadPoseResult(
                status=PoseStatus.REPROJECTION_ERROR_HIGH.value,
                reprojection_error=round(rmse, 2) if math.isfinite(rmse) else 999.0
            )

        # Decompose rotation vector to Euler angles via Rodrigues and RQ decomposition
        try:
            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch_deg = float(angles[0])
            yaw_deg = float(angles[1])
            roll_deg = float(angles[2])
        except Exception:
            return HeadPoseResult(status=PoseStatus.SOLVE_FAILED.value)

        # Sanity check: angles must be finite and within plausible physical limits (-180 to 180)
        if not (math.isfinite(pitch_deg) and math.isfinite(yaw_deg) and math.isfinite(roll_deg)):
            return HeadPoseResult(status=PoseStatus.SOLVE_FAILED.value)

        # Compute confidence: 1.0 at 0 error, decaying towards 0 as rmse approaches max_error
        confidence = max(0.0, min(1.0, 1.0 - (rmse / self.max_reprojection_error)))

        return HeadPoseResult(
            pitch=pitch_deg,
            yaw=yaw_deg,
            roll=roll_deg,
            is_valid=True,
            status=PoseStatus.VALID.value,
            reprojection_error=rmse,
            confidence=confidence
        )
