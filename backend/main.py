"""
AntiRHys Backend API
====================
FastAPI application for real-time driver drowsiness monitoring.
Streams video frames via WebSocket, analyzes facial landmarks via MediaPipe Face Mesh,
and tracks driver alertness using monotonic time and Eye Aspect Ratio (EAR).
"""

import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
import json
import math
import os
import time
from typing import Optional, Tuple
import uuid

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import numpy as np

from detector import (
    DrowsinessDetector,
    DetectionConfig,
    DetectionResult,
    DetectionState,
    TrackingState,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES
)
from head_pose import HEAD_POSE_LANDMARK_INDICES
from baseline_calibrator import CalibrationState
from session_logger import SessionLogger

# ---------------------------------------------------------------------------
# Configuration & Limits
# ---------------------------------------------------------------------------
MAX_MESSAGE_BYTES = 5 * 1024 * 1024       # 5 MB maximum raw message size
MAX_FRAME_WIDTH = 1920
MAX_FRAME_HEIGHT = 1080
MIN_FRAME_WIDTH = 64
MIN_FRAME_HEIGHT = 64
MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "10"))
MIN_FRAME_INTERVAL_SEC = 0.03            # Throttle processing to max ~30 FPS

# Safe default origins: restricted to local development unless explicitly configured
DEFAULT_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://testserver"
]

ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "")
if not ALLOWED_ORIGINS_ENV:
    ALLOWED_ORIGINS = DEFAULT_DEV_ORIGINS
elif ALLOWED_ORIGINS_ENV == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [orig.strip() for orig in ALLOWED_ORIGINS_ENV.split(",") if orig.strip()]

# Authentication token for session authorization
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()

# Global Session Logger
session_logger = SessionLogger(os.getenv("SESSION_LOG_PATH", "session_log.csv"))

def safe_log_event(**kwargs) -> None:
    """Safely log an event without allowing logging exceptions to crash detection sessions."""
    try:
        session_logger.log_event(**kwargs)
    except Exception as e:
        print(f"[SessionLogger] Unhandled logging error: {e}")

# Active connection counter
active_sessions = 0
active_sessions_lock = asyncio.Lock()

mp_face_mesh = mp.solutions.face_mesh


def create_isolated_face_mesh() -> mp.solutions.face_mesh.FaceMesh:
    """Create an isolated MediaPipe FaceMesh instance per WebSocket session."""
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AntiRHys backend initialized.")
    if not AUTH_TOKEN:
        print("[AntiRHys Security Notice] AUTH_TOKEN not set. Running in unauthenticated mode.")
    yield
    print("AntiRHys backend shutting down.")


app = FastAPI(
    title="AntiRHys Drowsiness Detection API",
    version="2.0.0",
    description="Safety-conscious driver drowsiness monitoring WebSocket backend.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials="*" not in ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for Render/Vercel and monitoring uptime."""
    return {
        "status": "healthy",
        "service": "AntiRHys Drowsiness Detection API",
        "version": "2.0.0",
        "active_sessions": active_sessions,
        "max_concurrent_sessions": MAX_CONCURRENT_SESSIONS,
        "auth_enabled": bool(AUTH_TOKEN),
        "timestamp": time.time()
    }


def is_origin_allowed(origin: Optional[str]) -> bool:
    """
    Validate WebSocket origin against configured CORS rules.
    Prevents Cross-Site WebSocket Hijacking (CSWSH) from browser contexts.
    Note: Origin validation alone does not authenticate the client.
    """
    if "*" in ALLOWED_ORIGINS:
        return True
    if not origin:
        # In strict configured production, missing origin is disallowed.
        # In default local development, allow missing origin for local scripts/tests.
        return not bool(ALLOWED_ORIGINS_ENV)
    norm_origin = origin.rstrip("/").lower()
    allowed_norm = [o.rstrip("/").lower() for o in ALLOWED_ORIGINS]
    return norm_origin in allowed_norm


def is_token_authorized(websocket: WebSocket) -> bool:
    """
    Validate client authorization token when AUTH_TOKEN is configured.
    Supports query parameter ?token=... or Authorization: Bearer <token>.
    """
    if not AUTH_TOKEN:
        return True

    # Check query param
    token = websocket.query_params.get("token")
    if token and token == AUTH_TOKEN:
        return True

    # Check Authorization header
    auth_header = websocket.headers.get("authorization")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1] == AUTH_TOKEN:
            return True

    return False


def process_frame_sync(
    face_mesh: mp.solutions.face_mesh.FaceMesh,
    img_bytes: bytes,
    detector: DrowsinessDetector,
    now: float
) -> Tuple[Optional[DetectionResult], Optional[str]]:
    """
    Synchronous image decode and MediaPipe inference routine.
    Intended to be executed in an asyncio thread worker to keep the event loop non-blocking.
    """
    try:
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return None, f"Image decoding error: {e}"

    if frame is None:
        return None, "Decoded image frame is null or corrupted."

    h, w, _ = frame.shape
    if w < MIN_FRAME_WIDTH or h < MIN_FRAME_HEIGHT:
        return None, f"Image resolution too small: {w}x{h} (minimum {MIN_FRAME_WIDTH}x{MIN_FRAME_HEIGHT})."
    if w > MAX_FRAME_WIDTH or h > MAX_FRAME_HEIGHT:
        # Downscale oversized frames to prevent memory thrashing
        scale = min(MAX_FRAME_WIDTH / w, MAX_FRAME_HEIGHT / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w, _ = frame.shape

    # MediaPipe FaceMesh processing
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
    except Exception as e:
        return None, f"MediaPipe inference failed: {e}"

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Extract 2D pixel coordinates for left and right eyes
        left_eye_points = []
        for idx in LEFT_EYE_INDICES:
            lm = face_landmarks.landmark[idx]
            left_eye_points.append((lm.x * w, lm.y * h))

        right_eye_points = []
        for idx in RIGHT_EYE_INDICES:
            lm = face_landmarks.landmark[idx]
            right_eye_points.append((lm.x * w, lm.y * h))

        # Extract normalized coordinates for frontend eye boundary rendering
        all_indices = LEFT_EYE_INDICES + RIGHT_EYE_INDICES
        normalized_landmarks = [
            {"x": face_landmarks.landmark[i].x, "y": face_landmarks.landmark[i].y}
            for i in all_indices
        ]

        # Extract 2D pixel coordinates for 3D head pose estimation
        head_pose_points = [
            (face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h)
            for idx in HEAD_POSE_LANDMARK_INDICES
        ]

        result = detector.process_frame_landmarks(
            has_face=True,
            left_eye_points=left_eye_points,
            right_eye_points=right_eye_points,
            normalized_landmarks=normalized_landmarks,
            now=now,
            frame_width=w,
            frame_height=h,
            head_pose_points=head_pose_points
        )
    else:
        # Face not detected
        result = detector.process_frame_landmarks(
            has_face=False,
            now=now,
            frame_width=w,
            frame_height=h
        )

    return result, None


@app.websocket("/ws/detect")
async def detect_fatigue(websocket: WebSocket):
    """
    WebSocket endpoint for real-time driver drowsiness monitoring.
    Guarantees active-session cleanup on all failure paths and preserves
    server-generated session ID integrity.
    """
    global active_sessions

    # 1. Origin verification before reserving connection capacity
    origin = websocket.headers.get("origin")
    if not is_origin_allowed(origin):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Origin not allowed")
        return

    # 2. Token authorization check
    if not is_token_authorized(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized token")
        return

    session_acquired = False
    face_mesh = None
    detector = None
    server_session_id = f"session_{datetime_stamp()}_{uuid.uuid4().hex[:6]}"

    try:
        # 3. Reserve concurrent session slot inside lock
        async with active_sessions_lock:
            if active_sessions >= MAX_CONCURRENT_SESSIONS:
                await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Server at max session capacity")
                return
            active_sessions += 1
            session_acquired = True

        # 4. Accept WebSocket connection
        await websocket.accept()

        detector = DrowsinessDetector(session_id=server_session_id)
        face_mesh = create_isolated_face_mesh()

        safe_log_event(
            session_id=server_session_id,
            event_type="SESSION_START",
            notes="Client connected and session started"
        )

        last_processed_monotonic = 0.0
        last_logged_detection_state = DetectionState.NORMAL
        last_logged_tracking_state = TrackingState.SEARCHING

        while True:
            raw_data = await websocket.receive_text()

            # Input validation: check message length ceiling
            if len(raw_data) > MAX_MESSAGE_BYTES:
                await websocket.send_json({
                    "error": "Payload exceeds maximum allowed size (5MB).",
                    "session_id": server_session_id
                })
                continue

            # Payload Schema Validation
            client_ts = None
            client_session_id = None
            frame_data = ""

            if raw_data.startswith("{") or raw_data.startswith("["):
                try:
                    payload = json.loads(raw_data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "error": "Malformed JSON payload.",
                        "session_id": server_session_id
                    })
                    continue

                if not isinstance(payload, dict):
                    await websocket.send_json({
                        "error": "Payload must be a JSON object.",
                        "session_id": server_session_id
                    })
                    continue

                # Validate 'client_timestamp'
                if "client_timestamp" in payload and payload["client_timestamp"] is not None:
                    ts_val = payload["client_timestamp"]
                    if not isinstance(ts_val, (int, float)) or not math.isfinite(ts_val) or ts_val < 0:
                        await websocket.send_json({
                            "error": "Invalid 'client_timestamp': must be a positive finite number.",
                            "session_id": server_session_id
                        })
                        continue
                    client_ts = float(ts_val)

                # Validate 'session_id' without allowing it to overwrite server_session_id
                if "session_id" in payload and payload["session_id"] is not None:
                    sid_val = payload["session_id"]
                    if not isinstance(sid_val, str) or len(sid_val) > 128:
                        await websocket.send_json({
                            "error": "Invalid 'session_id': must be a string up to 128 characters.",
                            "session_id": server_session_id
                        })
                        continue
                    client_session_id = sid_val

                # Optional calibration lifecycle commands
                if "command" in payload and isinstance(payload["command"], str):
                    cmd = payload["command"]
                    cmd_msg = ""
                    if cmd == "start_calibration":
                        driver_id = str(payload.get("driver_id", "driver"))[:64]
                        detector.start_calibration(driver_id=driver_id)
                        safe_log_event(
                            session_id=server_session_id,
                            event_type="CALIBRATION_START",
                            notes=f"Calibration started for {driver_id}"
                        )
                        cmd_msg = f"Calibration started for {driver_id}"
                    elif cmd == "abort_calibration":
                        detector.abort_calibration()
                        cmd_msg = "Calibration aborted"
                    elif cmd == "finalize_calibration":
                        detector.finalize_calibration()
                        cmd_msg = "Calibration finalized"
                    elif cmd == "clear_baseline":
                        detector.clear_baseline()
                        safe_log_event(
                            session_id=server_session_id,
                            event_type="BASELINE_CLEARED",
                            notes="Personalized baseline cleared"
                        )
                        cmd_msg = "Baseline cleared"

                    # If this is a control-only packet (no video frame attached), respond immediately
                    if "frame" not in payload:
                        now = time.monotonic()
                        perclos, p_valid, p_status, _, _ = detector.perclos_calculator.compute_perclos(now)
                        calib_state = detector.calibrator.state.value if detector.calibrator else "IDLE"
                        calib_reason = detector.calibrator.failure_reason.value if (detector.calibrator and detector.calibrator.state == CalibrationState.FAILED) else "NONE"
                        calib_prog = 0.0
                        if detector.calibrator:
                            if detector.calibrator.state == CalibrationState.COLLECTING and detector.calibrator._start_time:
                                dur = now - detector.calibrator._start_time
                                calib_prog = min(1.0, max(0.0, dur / detector.calibrator.config.target_duration_sec))
                            elif detector.calibrator.state == CalibrationState.COMPLETED:
                                calib_prog = 1.0

                        cmd_resp = {
                            "session_id": server_session_id,
                            "command": cmd,
                            "status": "success",
                            "status_message": cmd_msg or f"Command '{cmd}' processed.",
                            "detection_state": detector.detection_state.value,
                            "state": "Alert" if detector.detection_state == DetectionState.NORMAL else detector.detection_state.value,
                            "alarm_state": detector.alarm_state.value,
                            "tracking_state": detector.tracking_state.value,
                            "eye_state": detector.eye_state.value,
                            "ear": 0.0,
                            "left_ear": 0.0,
                            "right_ear": 0.0,
                            "blink_count": detector.blink_count,
                            "closure_duration": round(detector.closure_duration, 2),
                            "tracking_lost_duration": 0.0,
                            "landmarks": [],
                            "perclos": round(perclos, 3),
                            "perclos_valid": p_valid,
                            "perclos_status": p_status,
                            "perclos_window_duration": detector.config.perclos_window_sec,
                            "perclos_warning": False,
                            "baseline_active": (detector.baseline is not None and detector.baseline.is_valid),
                            "calibrated_ear_threshold": round(detector.config.ear_threshold, 3),
                            "calibration_state": calib_state,
                            "calibration_progress": round(calib_prog, 2),
                            "calibration_failure_reason": calib_reason,
                            "pitch": 0.0,
                            "yaw": 0.0,
                            "roll": 0.0,
                            "head_pose_valid": False,
                            "head_pose_status": "UNAVAILABLE",
                            "head_pose_reprojection_error": 0.0,
                            "head_pose_confidence": 0.0,
                        }
                        if client_session_id is not None:
                            cmd_resp["client_session_id"] = client_session_id
                        if client_ts is not None:
                            cmd_resp["client_timestamp"] = client_ts
                        await websocket.send_json(cmd_resp)
                        continue

                # Validate 'frame' field for image inference
                if "frame" not in payload or not isinstance(payload["frame"], str):
                    await websocket.send_json({
                        "error": "Missing or invalid 'frame' field: must be a Base64 string.",
                        "session_id": server_session_id
                    })
                    continue

                frame_data = payload["frame"]
                if len(frame_data) < 10:
                    await websocket.send_json({
                        "error": "'frame' field is too short to be valid image data.",
                        "session_id": server_session_id
                    })
                    continue
            else:
                # Raw Base64 string mode
                frame_data = raw_data
                if len(frame_data) < 10:
                    await websocket.send_json({
                        "error": "Raw frame payload too short.",
                        "session_id": server_session_id
                    })
                    continue

            # Rate-limiting check: throttle frames arriving faster than interval
            now = time.monotonic()
            if (now - last_processed_monotonic) < MIN_FRAME_INTERVAL_SEC:
                # Silently discard excessive frames to avoid queuing
                continue

            # Extract pure base64 bytes
            if "," in frame_data:
                base64_str = frame_data.split(",", 1)[1]
            else:
                base64_str = frame_data

            try:
                img_bytes = base64.b64decode(base64_str, validate=True)
            except (binascii.Error, ValueError):
                err_res = detector.process_frame_error("Invalid base64 encoding.", now=now)
                resp = err_res.to_dict()
                resp["error"] = "Invalid base64 encoding."
                resp["session_id"] = server_session_id
                resp["state"] = err_res.detection_state.value
                await websocket.send_json(resp)
                continue

            last_processed_monotonic = now

            # Offload heavy synchronous inference to asyncio thread pool
            result, err = await asyncio.to_thread(
                process_frame_sync,
                face_mesh,
                img_bytes,
                detector,
                now
            )

            if err is not None or result is None:
                err_res = detector.process_frame_error(err or "Processing error", now=now)
                resp = err_res.to_dict()
                resp["error"] = err or "Processing error"
                resp["session_id"] = server_session_id
                resp["state"] = err_res.detection_state.value
                await websocket.send_json(resp)
                continue

            # Check for significant state transitions to log
            if (result.detection_state != last_logged_detection_state or
                    result.tracking_state != last_logged_tracking_state):

                event_type = "STATE_CHANGE"
                if result.detection_state == DetectionState.POSSIBLE_DROWSINESS:
                    event_type = "DROWSINESS_ALERT"
                elif (last_logged_detection_state == DetectionState.POSSIBLE_DROWSINESS and
                      result.detection_state == DetectionState.NORMAL):
                    event_type = "RECOVERY"
                elif result.tracking_state == TrackingState.LOST:
                    event_type = "TRACKING_LOST"
                elif (last_logged_tracking_state == TrackingState.LOST and
                      result.tracking_state == TrackingState.ACTIVE):
                    event_type = "TRACKING_RESTORED"

                safe_log_event(
                    session_id=server_session_id,
                    event_type=event_type,
                    detection_state=result.detection_state.value,
                    eye_state=result.eye_state.value,
                    tracking_state=result.tracking_state.value,
                    ear=result.ear,
                    blink_count=result.blink_count,
                    closure_duration=result.closure_duration,
                    perclos=result.perclos,
                    perclos_status=result.perclos_status,
                    notes=result.status_message
                )

                last_logged_detection_state = result.detection_state
                last_logged_tracking_state = result.tracking_state

            # Construct structured response
            response = result.to_dict()
            response["session_id"] = server_session_id
            if client_session_id is not None:
                response["client_session_id"] = client_session_id
            if client_ts is not None:
                response["client_timestamp"] = client_ts

            response["state"] = "Drowsy" if result.detection_state == DetectionState.POSSIBLE_DROWSINESS else (
                "Alert" if result.detection_state == DetectionState.NORMAL else result.detection_state.value
            )

            await websocket.send_json(response)

    except WebSocketDisconnect:
        if detector is not None:
            safe_log_event(
                session_id=server_session_id,
                event_type="SESSION_END",
                detection_state=detector.detection_state.value,
                notes="Client disconnected gracefully"
            )
    except Exception as e:
        safe_log_event(
            session_id=server_session_id,
            event_type="ERROR",
            notes=f"Unexpected exception: {e}"
        )
    finally:
        # Cleanup isolated MediaPipe instance
        if face_mesh is not None:
            try:
                face_mesh.close()
            except Exception:
                pass
        if detector is not None:
            detector.stop_session()

        # Guarantee active session count release on all failure paths
        if session_acquired:
            async with active_sessions_lock:
                active_sessions = max(0, active_sessions - 1)


def datetime_stamp() -> str:
    """Generate compact datetime stamp for session ids."""
    return time.strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
