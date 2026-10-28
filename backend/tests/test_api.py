"""
Integration tests for AntiRHys FastAPI application and WebSocket endpoints.
"""

import base64
import json
import numpy as np
import cv2
import pytest
from starlette.websockets import WebSocketDisconnect
from fastapi.testclient import TestClient
import main
from main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_base64_frame():
    """Generate a valid 320x240 blank test image in JPEG Base64 format."""
    blank_img = np.zeros((240, 320, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', blank_img)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "active_sessions" in data
    assert "auth_enabled" in data
    assert "environment" in data


def test_websocket_detect_valid_frame(client, sample_base64_frame):
    with client.websocket_connect("/ws/detect") as websocket:
        websocket.send_text(sample_base64_frame)
        response = websocket.receive_json()

        assert "session_id" in response
        assert response["session_id"].startswith("session_")
        assert "state" in response
        assert "tracking_state" in response
        assert "ear" in response
        assert "landmarks" in response


def test_session_id_consistency_server_authority(client, sample_base64_frame):
    """
    Verify client-provided session IDs do NOT overwrite server internal session IDs.
    Client session ID is echoed back as client_session_id for correlation only.
    """
    with client.websocket_connect("/ws/detect") as websocket:
        msg = {
            "session_id": "client_arbitrary_id_12345",
            "client_timestamp": 1700000000123,
            "frame": sample_base64_frame
        }
        websocket.send_text(json.dumps(msg))
        response = websocket.receive_json()

        # Authoritative server session_id is preserved
        assert response["session_id"] != "client_arbitrary_id_12345"
        assert response["session_id"].startswith("session_")
        # Client session_id is correlated separately
        assert response.get("client_session_id") == "client_arbitrary_id_12345"
        assert response.get("client_timestamp") == 1700000000123


def test_active_session_counter_release_on_disconnect(client, sample_base64_frame):
    """
    Verify active_sessions counter increments during connection and cleanly decrements
    to prior baseline on disconnect.
    """
    initial_active = main.active_sessions
    with client.websocket_connect("/ws/detect") as websocket:
        websocket.send_text(sample_base64_frame)
        _ = websocket.receive_json()
        assert main.active_sessions == initial_active + 1

    # After exiting with block, socket is closed
    assert main.active_sessions == initial_active


def test_websocket_malformed_base64(client):
    with client.websocket_connect("/ws/detect") as websocket:
        websocket.send_text("data:image/jpeg;base64,not_valid_base64!@#$%")
        response = websocket.receive_json()
        assert "error" in response
        assert "base64" in response["error"].lower()


def test_websocket_oversized_payload(client):
    with client.websocket_connect("/ws/detect") as websocket:
        # Create a string larger than 5MB
        huge_payload = "A" * (6 * 1024 * 1024)
        websocket.send_text(huge_payload)
        response = websocket.receive_json()
        assert "error" in response
        assert "exceeds" in response["error"].lower()


def test_payload_schema_rejects_non_dict_json(client):
    with client.websocket_connect("/ws/detect") as websocket:
        websocket.send_text("[1, 2, 3]")
        response = websocket.receive_json()
        assert "error" in response
        assert "JSON object" in response["error"]


def test_payload_schema_rejects_missing_or_invalid_frame(client):
    with client.websocket_connect("/ws/detect") as websocket:
        # Frame missing
        websocket.send_text(json.dumps({"session_id": "abc"}))
        res1 = websocket.receive_json()
        assert "error" in res1
        assert "frame" in res1["error"].lower()

        # Frame is integer, not string
        websocket.send_text(json.dumps({"frame": 12345}))
        res2 = websocket.receive_json()
        assert "error" in res2
        assert "frame" in res2["error"].lower()


def test_payload_schema_rejects_invalid_client_timestamp(client, sample_base64_frame):
    with client.websocket_connect("/ws/detect") as websocket:
        # Timestamp is string
        websocket.send_text(json.dumps({"frame": sample_base64_frame, "client_timestamp": "invalid_ts"}))
        res1 = websocket.receive_json()
        assert "error" in res1
        assert "client_timestamp" in res1["error"]

        # Timestamp is negative
        websocket.send_text(json.dumps({"frame": sample_base64_frame, "client_timestamp": -50.0}))
        res2 = websocket.receive_json()
        assert "error" in res2
        assert "client_timestamp" in res2["error"]


def test_payload_schema_rejects_invalid_session_id(client, sample_base64_frame):
    with client.websocket_connect("/ws/detect") as websocket:
        # Oversized session_id (>128 chars)
        websocket.send_text(json.dumps({"frame": sample_base64_frame, "session_id": "x" * 200}))
        res = websocket.receive_json()
        assert "error" in res
        assert "session_id" in res["error"]


def test_token_authentication_when_configured(client, sample_base64_frame):
    """
    Test token authorization when AUTH_TOKEN is active.
    """
    original_token = main.AUTH_TOKEN
    try:
        main.AUTH_TOKEN = "secret_test_token_999"

        # 1. Unauthenticated connection rejected
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/detect"):
                pass
        assert exc_info.value.code == 1008

        # 2. Invalid token rejected
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/detect?token=wrong_token"):
                pass
        assert exc_info.value.code == 1008

        # 3. Valid token query param accepted
        with client.websocket_connect("/ws/detect?token=secret_test_token_999") as ws:
            ws.send_text(sample_base64_frame)
            res = ws.receive_json()
            assert "session_id" in res

        # 4. Valid token in Authorization Bearer header accepted
        with client.websocket_connect(
            "/ws/detect",
            headers={"authorization": "Bearer secret_test_token_999"}
        ) as ws:
            ws.send_text(sample_base64_frame)
            res = ws.receive_json()
            assert "session_id" in res

    finally:
        main.AUTH_TOKEN = original_token


def test_origin_validation_when_restricted(client):
    """
    Test that origin validation rejects unlisted origins when restricted.
    """
    original_origins = main.ALLOWED_ORIGINS
    original_env = main.ALLOWED_ORIGINS_ENV
    try:
        main.ALLOWED_ORIGINS_ENV = "https://trusted.example.com"
        main.ALLOWED_ORIGINS = ["https://trusted.example.com"]

        # 1. Unlisted origin rejected
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/detect", headers={"origin": "https://evil.example.com"}):
                pass
        assert exc_info.value.code == 1008

        # 2. Missing origin header when restricted rejected
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/detect"):
                pass
        assert exc_info.value.code == 1008

        # 3. Trusted origin accepted
        with client.websocket_connect("/ws/detect", headers={"origin": "https://trusted.example.com"}):
            pass

    finally:
        main.ALLOWED_ORIGINS = original_origins
        main.ALLOWED_ORIGINS_ENV = original_env

def test_origin_validation_production_trusted_vs_untrusted_vercel():
    """
    Test that production mode strictly allows the official deployment origin
    and rejects arbitrary *.vercel.app hostnames (preventing CSWSH).
    """
    orig_prod = main.IS_PRODUCTION
    orig_origins = main.ALLOWED_ORIGINS
    try:
        main.IS_PRODUCTION = True
        main.ALLOWED_ORIGINS = [main.OFFICIAL_PRODUCTION_ORIGIN]

        # Official production origin allowed (with or without trailing slash)
        assert main.is_origin_allowed("https://antirhys.vercel.app") is True
        assert main.is_origin_allowed("https://antirhys.vercel.app/") is True

        # Arbitrary / untrusted Vercel tenant origins strictly REJECTED in production
        assert main.is_origin_allowed("https://attacker.vercel.app") is False
        assert main.is_origin_allowed("https://unrelated-subdomain.vercel.app") is False
        assert main.is_origin_allowed("https://evil.com") is False

        # Missing origin strictly REJECTED in production mode
        assert main.is_origin_allowed(None) is False
        assert main.is_origin_allowed("") is False
    finally:
        main.IS_PRODUCTION = orig_prod
        main.ALLOWED_ORIGINS = orig_origins


def test_origin_validation_development_vs_production_mode():
    """
    Test behavior difference between development mode and production mode.
    """
    orig_prod = main.IS_PRODUCTION
    orig_origins = main.ALLOWED_ORIGINS
    orig_env = main.ALLOWED_ORIGINS_ENV
    try:
        # Development mode without explicit env: allows localhost and missing origin
        main.IS_PRODUCTION = False
        main.ALLOWED_ORIGINS_ENV = ""
        main.ALLOWED_ORIGINS = main.DEV_DEFAULT_ORIGINS

        assert main.is_origin_allowed(None) is True
        assert main.is_origin_allowed("http://localhost:5173") is True
        assert main.is_origin_allowed("https://antirhys.vercel.app") is True
        assert main.is_origin_allowed("https://attacker.vercel.app") is False

        # Production mode without env: disallows localhost and missing origin by default
        main.IS_PRODUCTION = True
        main.ALLOWED_ORIGINS = main.PRODUCTION_DEFAULT_ORIGINS

        assert main.is_origin_allowed(None) is False
        assert main.is_origin_allowed("http://localhost:5173") is False
        assert main.is_origin_allowed("https://antirhys.vercel.app") is True
    finally:
        main.IS_PRODUCTION = orig_prod
        main.ALLOWED_ORIGINS = orig_origins
        main.ALLOWED_ORIGINS_ENV = orig_env


def test_parse_allowed_origins_sanitization():
    """
    Test origin parsing and sanitization function across valid and malformed inputs.
    """
    # Trims whitespace, lowercases, and strips trailing slashes
    raw = " https://antirhys.vercel.app/ , http://localhost:5173/ , https://ANOTHER.DOMAIN.COM/ "
    parsed = main.parse_allowed_origins(raw)
    assert parsed == [
        "https://antirhys.vercel.app",
        "http://localhost:5173",
        "https://another.domain.com",
    ]

    # Empty string or commas only
    assert main.parse_allowed_origins("") == []
    assert main.parse_allowed_origins(" , , ") == []

    # Wildcard origin
    assert main.parse_allowed_origins(" * ") == ["*"]


def test_origin_validation_wildcard_configuration():
    """Test that explicit wildcard allows any origin."""
    orig_origins = main.ALLOWED_ORIGINS
    try:
        main.ALLOWED_ORIGINS = ["*"]
        assert main.is_origin_allowed("https://any-site.com") is True
        assert main.is_origin_allowed(None) is True
    finally:
        main.ALLOWED_ORIGINS = orig_origins


def test_authentication_disabled_by_default(client, sample_base64_frame):
    """Test that when AUTH_TOKEN is not configured, connections proceed without tokens."""
    orig_token = main.AUTH_TOKEN
    try:
        main.AUTH_TOKEN = ""
        with client.websocket_connect("/ws/detect") as ws:
            ws.send_text(sample_base64_frame)
            res = ws.receive_json()
            assert "session_id" in res
    finally:
        main.AUTH_TOKEN = orig_token


def test_websocket_reconnection_lifecycle(client, sample_base64_frame):
    """
    Verify complete connection, frame processing, disconnect, session cleanup,
    and clean immediate reconnection.
    """
    assert main.active_sessions == 0

    # 1. First session connects
    with client.websocket_connect("/ws/detect") as ws1:
        assert main.active_sessions == 1
        ws1.send_text(sample_base64_frame)
        res1 = ws1.receive_json()
        assert "session_id" in res1
        session_id_1 = res1["session_id"]
        assert main.active_sessions == 1

    # First session disconnected - active_sessions decremented
    assert main.active_sessions == 0

    # 2. Reconnect immediately - creates new independent session
    with client.websocket_connect("/ws/detect") as ws2:
        assert main.active_sessions == 1
        ws2.send_text(sample_base64_frame)
        res2 = ws2.receive_json()
        assert "session_id" in res2
        session_id_2 = res2["session_id"]
        assert session_id_1 != session_id_2
        assert main.active_sessions == 1

    # Clean teardown
    assert main.active_sessions == 0
