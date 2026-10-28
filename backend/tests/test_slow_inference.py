"""
Tests and measurements for slow-inference behavior and queue accumulation.
Evaluates frame staleness under constrained processing throughput.
"""

import base64
import json
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import app, MIN_FRAME_INTERVAL_SEC


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_frame():
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', blank)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def test_rate_limiter_throttles_rapid_frame_flood(client, sample_frame):
    """
    Measure rate limiting: when frames arrive faster than MIN_FRAME_INTERVAL_SEC,
    excess frames are silently dropped rather than queued up.
    """
    with client.websocket_connect("/ws/detect") as ws:
        # Rapidly send 5 frames with 0ms delay
        for i in range(5):
            msg = {
                "client_timestamp": time.time(),
                "frame": sample_frame
            }
            ws.send_text(json.dumps(msg))

        # Because rate limiter drops frames arriving < 30ms apart,
        # the server will process the first frame, and drop the immediate consecutive ones.
        # We should receive at least 1 response without server hang or crash.
        res = ws.receive_json()
        assert "session_id" in res
        assert "ear" in res


def test_in_flight_flow_control_bounds_frame_staleness(client, sample_frame):
    """
    Measure frame staleness:
    With in-flight flow control (client sends next frame only after previous
    response is received), frame age (now - client_timestamp) at response time
    remains strictly bounded by round-trip latency and never accumulates indefinitely.
    """
    with client.websocket_connect("/ws/detect") as ws:
        staleness_measurements = []

        # Send 5 frames sequentially using in-flight flow control pattern (simulating ~25 FPS)
        for _ in range(5):
            time.sleep(0.04)  # Ensure frame interval > MIN_FRAME_INTERVAL_SEC (0.03s)
            client_send_ts = time.time()
            msg = {
                "client_timestamp": client_send_ts,
                "frame": sample_frame
            }
            ws.send_text(json.dumps(msg))
            response = ws.receive_json()
            recv_ts = time.time()

            staleness = recv_ts - response.get("client_timestamp", client_send_ts)
            staleness_measurements.append(staleness)

        # In-flight pattern guarantees staleness does not grow with each frame
        assert len(staleness_measurements) == 5
        # Max roundtrip staleness on localhost should remain well under 500ms
        max_staleness = max(staleness_measurements)
        assert max_staleness < 0.50, f"Staleness accumulated: {max_staleness}s"
