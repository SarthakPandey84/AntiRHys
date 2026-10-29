import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Video, VideoOff, RefreshCw, AlertCircle } from 'lucide-react';
import type { DetectionState, ConnectionStatus, CameraStatus } from '../types';
import { getWebSocketUrl } from '../config';
import { audioAlertService } from '../utils/audioAlert';

interface CameraStreamProps {
  onDetectionUpdate: React.Dispatch<React.SetStateAction<DetectionState>>;
  onConnectionChange?: (status: ConnectionStatus) => void;
  onCameraChange?: (status: CameraStatus, errorMsg?: string) => void;
  landmarks: { x: number; y: number }[];
  onRegisterSendCommand?: (sendCommand: (command: string, extra?: Record<string, unknown>) => boolean) => void;
}

const MAX_RECONNECT_ATTEMPTS = 20;
const STALE_TIMEOUT_MS = 2500;
const MAX_BUFFERED_BYTES = 64 * 1024; // 64 KB buffer threshold

const CameraStream: React.FC<CameraStreamProps> = ({
  onDetectionUpdate,
  onConnectionChange,
  onCameraChange,
  landmarks,
  onRegisterSendCommand,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const extractCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureIntervalRef = useRef<number | null>(null);
  const staleTimerRef = useRef<number | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const activeStreamRef = useRef<MediaStream | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const isStreamingRef = useRef<boolean>(false);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('idle');
  const [cameraError, setCameraError] = useState<string | null>(null);

  const clientSessionIdRef = useRef<string>('');
  const serverSessionIdRef = useRef<string>('');
  const lastResponseTimeRef = useRef<number>(0);
  const lastProcessedClientTsRef = useRef<number>(0);
  const reconnectAttemptsRef = useRef<number>(0);
  const connectWebSocketRef = useRef<() => void>(() => {});
  const isStartingCameraRef = useRef<boolean>(false);
  const isMountedRef = useRef<boolean>(true);
  const isFrameInFlightRef = useRef<boolean>(false);
  const lastFrameSentTimeRef = useRef<number>(0);

  const updateConnectionStatus = useCallback(
    (status: ConnectionStatus) => {
      setConnectionStatus(status);
      onConnectionChange?.(status);
    },
    [onConnectionChange]
  );

  const updateCameraStatus = useCallback(
    (status: CameraStatus, errorMsg?: string) => {
      setCameraStatus(status);
      setCameraError(errorMsg || null);
      onCameraChange?.(status, errorMsg);
    },
    [onCameraChange]
  );

  // Cleanly close WebSocket
  const closeWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current !== null) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    isFrameInFlightRef.current = false;
    if (wsRef.current) {
      try {
        wsRef.current.onopen = null;
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        wsRef.current.onmessage = null;
        if (
          wsRef.current.readyState === WebSocket.OPEN ||
          wsRef.current.readyState === WebSocket.CONNECTING
        ) {
          wsRef.current.close(1000, 'Session stopped');
        }
      } catch (err) {
        console.warn('Error closing WebSocket:', err);
      }
      wsRef.current = null;
    }
    updateConnectionStatus('disconnected');
  }, [updateConnectionStatus]);

  // Connect WebSocket with bounded exponential backoff
  const connectWebSocket = useCallback(() => {
    closeWebSocket();

    const wsUrl = getWebSocketUrl();
    updateConnectionStatus(
      reconnectAttemptsRef.current > 0 ? 'reconnecting' : 'connecting'
    );

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectAttemptsRef.current = 0;
        updateConnectionStatus('connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          // If server echoed a client_session_id from an older session, ignore it
          if (
            data.client_session_id &&
            clientSessionIdRef.current &&
            data.client_session_id !== clientSessionIdRef.current
          ) {
            return;
          }

          // Guard against out-of-order older responses
          if (data.client_timestamp && data.client_timestamp < lastProcessedClientTsRef.current) {
            return;
          }
          if (data.client_timestamp) {
            lastProcessedClientTsRef.current = data.client_timestamp;
          }

          lastResponseTimeRef.current = Date.now();
          isFrameInFlightRef.current = false;

          // Store authoritative server session ID
          if (data.session_id) {
            serverSessionIdRef.current = data.session_id;
          }

          // If backend returned an error, update message without wiping state
          if (data.error) {
            onDetectionUpdate((prev: DetectionState) => ({
              ...prev,
              error: data.error,
              status_message: data.error,
              last_received_time: Date.now(),
              is_stale: false,
            }));
            return;
          }

          // If this is a control-only response, merge without clearing live landmarks/ear
          if (data.command && (!data.landmarks || data.landmarks.length === 0)) {
            onDetectionUpdate((prev: DetectionState) => ({
              ...prev,
              command: data.command,
              status: data.status,
              status_message: data.status_message ?? prev.status_message,
              calibration_state: data.calibration_state ?? prev.calibration_state,
              calibration_progress: data.calibration_progress ?? prev.calibration_progress,
              calibration_failure_reason: data.calibration_failure_reason ?? prev.calibration_failure_reason,
              baseline_active: data.baseline_active ?? prev.baseline_active,
              calibrated_ear_threshold: data.calibrated_ear_threshold ?? prev.calibrated_ear_threshold,
              pitch: data.pitch !== undefined && data.pitch !== 0 ? data.pitch : prev.pitch,
              yaw: data.yaw !== undefined && data.yaw !== 0 ? data.yaw : prev.yaw,
              roll: data.roll !== undefined && data.roll !== 0 ? data.roll : prev.roll,
              head_pose_valid: data.head_pose_valid ?? prev.head_pose_valid,
              head_pose_status: data.head_pose_status ?? prev.head_pose_status,
              last_received_time: Date.now(),
              is_stale: false,
            }));
          } else {
            onDetectionUpdate({
              ...data,
              last_received_time: Date.now(),
              is_stale: false,
            });
          }
        } catch (err) {
          console.error('Error parsing backend WS message:', err);
        }
      };

      ws.onerror = (err) => {
        console.warn('WebSocket connection error:', err);
        isFrameInFlightRef.current = false;
      };

      ws.onclose = () => {
        wsRef.current = null;
        isFrameInFlightRef.current = false;

        // Reset calibration if connection drops mid-calibration
        onDetectionUpdate((prev: DetectionState) => {
          if (prev.calibration_state === 'COLLECTING') {
            return {
              ...prev,
              calibration_state: 'FAILED',
              calibration_failure_reason: 'CONNECTION_DROPPED',
              calibration_progress: 0,
              status_message: 'Calibration interrupted by connection drop. Please reconnect and retry.',
              is_stale: true,
            };
          }
          return prev;
        });

        if (!isStreamingRef.current) {
          updateConnectionStatus('disconnected');
          return;
        }

        // Attempt reconnection if streaming is active and attempts under limit
        if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
          const delay = Math.min(
            1000 * Math.pow(1.3, reconnectAttemptsRef.current),
            4000
          );
          reconnectAttemptsRef.current += 1;
          updateConnectionStatus('reconnecting');
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connectWebSocketRef.current();
          }, delay);
        } else {
          updateConnectionStatus('error');
          console.error('Max WebSocket reconnect attempts reached.');
          onDetectionUpdate((prev: DetectionState) => ({
            ...prev,
            status_message: 'Connection to detection server failed after multiple attempts. Ensure backend is running.',
            is_stale: true,
          }));
        }
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      updateConnectionStatus('error');
    }
  }, [closeWebSocket, onDetectionUpdate, updateConnectionStatus]);

  useEffect(() => {
    connectWebSocketRef.current = connectWebSocket;
  }, [connectWebSocket]);

  // Expose command sender to parent
  const sendCommand = useCallback((command: string, extra: Record<string, unknown> = {}): boolean => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return false;
    }
    try {
      const payload = {
        command,
        session_id: clientSessionIdRef.current || undefined,
        client_timestamp: Date.now(),
        ...extra,
      };
      ws.send(JSON.stringify(payload));
      return true;
    } catch (err) {
      console.error('Failed to send WebSocket command:', err);
      return false;
    }
  }, []);

  useEffect(() => {
    onRegisterSendCommand?.(sendCommand);
    return () => {
      onRegisterSendCommand?.(null as unknown as ((command: string, extra?: Record<string, unknown>) => boolean));
    };
  }, [onRegisterSendCommand, sendCommand]);

  // Watchdog timer to flag stale detection frames
  useEffect(() => {
    if (isStreaming) {
      staleTimerRef.current = window.setInterval(() => {
        if (
          lastResponseTimeRef.current > 0 &&
          Date.now() - lastResponseTimeRef.current > STALE_TIMEOUT_MS
        ) {
          onDetectionUpdate((prev: DetectionState) => ({
            ...prev,
            ear: 0,
            state: 'STALE',
            detection_state: 'STALE',
            tracking_state: 'LOST',
            eye_state: 'UNKNOWN',
            blink_count: 0,
            landmarks: [],
            is_stale: true,
            status_message: 'Detection response is stale. Backend may be lagging or disconnected.',
            perclos: 0,
            perclos_valid: false,
            perclos_status: 'INSUFFICIENT_DATA',
            perclos_window_duration: 60,
            perclos_warning: false,
          }));
        }
      }, 1000);
    } else {
      if (staleTimerRef.current !== null) {
        clearInterval(staleTimerRef.current);
        staleTimerRef.current = null;
      }
    }

    return () => {
      if (staleTimerRef.current !== null) {
        clearInterval(staleTimerRef.current);
      }
    };
  }, [isStreaming, onDetectionUpdate]);

  // Frame Capture Loop
  useEffect(() => {
    if (isStreaming) {
      captureIntervalRef.current = window.setInterval(() => {
        const video = videoRef.current;
        const canvas = extractCanvasRef.current;
        const ws = wsRef.current;

        if (
          video &&
          canvas &&
          ws &&
          ws.readyState === WebSocket.OPEN &&
          video.videoWidth > 0 &&
          video.videoHeight > 0
        ) {
          // Check WebSocket backpressure: drop frame if socket buffer is congested
          if (ws.bufferedAmount > MAX_BUFFERED_BYTES) {
            return;
          }

          // Slow-inference in-flight guard: do not queue frames if backend is still processing
          // If in-flight for > 400ms, assume dropped response and allow sending next frame
          if (isFrameInFlightRef.current && (Date.now() - lastFrameSentTimeRef.current < 400)) {
            return;
          }

          const ctx = canvas.getContext('2d');
          if (ctx) {
            // Constrain mobile raster resolution to max dimension 640 for CPU & battery efficiency
            const maxDim = 640;
            const maxCurrent = Math.max(video.videoWidth, video.videoHeight);
            const scale = maxCurrent > maxDim ? maxDim / maxCurrent : 1;
            canvas.width = Math.round(video.videoWidth * scale);
            canvas.height = Math.round(video.videoHeight * scale);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const base64Data = canvas.toDataURL('image/jpeg', 0.65);
            const message = JSON.stringify({
              session_id: clientSessionIdRef.current,
              client_timestamp: Date.now(),
              frame: base64Data,
            });

            isFrameInFlightRef.current = true;
            lastFrameSentTimeRef.current = Date.now();
            ws.send(message);
          }
        }
      }, 100); // 10 FPS
    } else {
      if (captureIntervalRef.current !== null) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
    }

    return () => {
      if (captureIntervalRef.current !== null) {
        clearInterval(captureIntervalRef.current);
      }
    };
  }, [isStreaming]);

  // Stop Camera and Session
  const stopCamera = useCallback(() => {
    isStartingCameraRef.current = false;
    isFrameInFlightRef.current = false;

    // Stop all media tracks
    if (activeStreamRef.current) {
      activeStreamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // Ignore track stop error
        }
      });
      activeStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Stop intervals
    if (captureIntervalRef.current !== null) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    if (staleTimerRef.current !== null) {
      clearInterval(staleTimerRef.current);
      staleTimerRef.current = null;
    }

    isStreamingRef.current = false;
    setIsStreaming(false);
    isFrameInFlightRef.current = false;
    lastProcessedClientTsRef.current = 0;
    clientSessionIdRef.current = '';
    serverSessionIdRef.current = '';
    updateCameraStatus('stopped');
    closeWebSocket();

    // Clear overlay
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      }
    }

    // Reset detection state
    onDetectionUpdate({
      ear: 0,
      state: 'IDLE',
      detection_state: 'IDLE',
      tracking_state: 'LOST',
      eye_state: 'UNKNOWN',
      blink_count: 0,
      landmarks: [],
      status_message: 'Monitoring session stopped.',
      perclos: 0,
      perclos_valid: false,
      perclos_status: 'CALCULATING',
      perclos_window_duration: 60,
      perclos_warning: false,
      calibration_state: 'IDLE',
      calibration_progress: 0,
      baseline_active: false,
      pitch: 0,
      yaw: 0,
      roll: 0,
      head_pose_valid: false,
      head_pose_status: 'UNAVAILABLE',
    });
  }, [closeWebSocket, onDetectionUpdate, updateCameraStatus]);

  // Start Camera and Session
  const startCamera = useCallback(async () => {
    setCameraError(null);
    updateCameraStatus('requesting');
    isStartingCameraRef.current = true;

    // Create fresh session ID and reset metrics
    const newSessionId = `client_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    clientSessionIdRef.current = newSessionId;
    serverSessionIdRef.current = '';
    reconnectAttemptsRef.current = 0;
    lastResponseTimeRef.current = 0;

    // Reset parent detection state
    onDetectionUpdate({
      ear: 0,
      state: 'CONNECTING',
      detection_state: 'CONNECTING',
      tracking_state: 'SEARCHING',
      eye_state: 'UNKNOWN',
      blink_count: 0,
      landmarks: [],
      session_id: newSessionId,
      status_message: 'Starting camera and connecting to detection service...',
      perclos: 0,
      perclos_valid: false,
      perclos_status: 'CALCULATING',
      perclos_window_duration: 60,
      perclos_warning: false,
      calibration_state: 'IDLE',
      calibration_progress: 0,
      baseline_active: false,
      pitch: 0,
      yaw: 0,
      roll: 0,
      head_pose_valid: false,
      head_pose_status: 'UNAVAILABLE',
    });

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Webcam access is not supported by your browser.');
      }

      // Unlock mobile audio context under the user-gesture click
      audioAlertService.initAudio().catch(() => {});

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640, max: 1280 },
          height: { ideal: 480, max: 720 },
          facingMode: { ideal: 'user' },
        },
      });

      // Cancellation check: user stopped camera or component unmounted while awaiting prompt
      if (!isMountedRef.current || !isStartingCameraRef.current) {
        stream.getTracks().forEach((track) => {
          try {
            track.stop();
          } catch {
            // ignore
          }
        });
        return;
      }
      isStartingCameraRef.current = false;

      // Listen for mobile track interruption (e.g. backgrounding, incoming call)
      stream.getVideoTracks().forEach((track) => {
        track.onended = () => {
          if (isMountedRef.current && isStreamingRef.current) {
            console.warn('Camera track ended by system/OS.');
            stopCamera();
            setCameraError('Camera stream was interrupted by the system or another application.');
            updateCameraStatus('error', 'Camera stream interrupted');
          }
        };
      });

      activeStreamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch((err) => {
            console.warn('Video playback warning:', err);
          });
        };
      }

      updateCameraStatus('active');
      isStreamingRef.current = true;
      setIsStreaming(true);
      lastProcessedClientTsRef.current = 0;
      connectWebSocket();
    } catch (err: unknown) {
      let msg = 'Could not access webcam. Please check your camera settings.';
      const e = err as { name?: string; message?: string };
      if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
        msg = 'Camera permission denied. Please allow camera permissions in your browser.';
      } else if (e.name === 'NotFoundError' || e.name === 'DevicesNotFoundError') {
        msg = 'No camera device detected on your system.';
      } else if (e.name === 'NotReadableError' || e.name === 'TrackStartError') {
        msg = 'Camera is already in use by another application or tab.';
      } else if (e.message) {
        msg = e.message;
      }

      console.error('Camera access error:', err);
      updateCameraStatus('denied', msg);
      setIsStreaming(false);
      onDetectionUpdate({
        ear: 0,
        state: 'UNAVAILABLE',
        detection_state: 'UNAVAILABLE',
        tracking_state: 'LOST',
        eye_state: 'UNKNOWN',
        blink_count: 0,
        landmarks: [],
        status_message: msg,
        perclos: 0,
        perclos_valid: false,
        perclos_status: 'UNAVAILABLE',
        perclos_window_duration: 60,
        perclos_warning: false,
      });
    }
  }, [connectWebSocket, onDetectionUpdate, stopCamera, updateCameraStatus]);

  // Clean up all resources on component unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      isStartingCameraRef.current = false;
      stopCamera();
    };
  }, [stopCamera]);

  // Draw eye landmarks overlay
  useEffect(() => {
    if (!overlayCanvasRef.current || !videoRef.current) return;
    const canvas = overlayCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = videoRef.current.clientWidth || 640;
    canvas.height = videoRef.current.clientHeight || 480;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (landmarks && landmarks.length > 0 && isStreaming) {
      ctx.fillStyle = '#22c55e';
      landmarks.forEach((point) => {
        const x = point.x * canvas.width;
        const y = point.y * canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
  }, [landmarks, isStreaming]);

  return (
    <div className="stream-container glass-panel">
      <div className="video-wrapper">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="video-feed"
          aria-label="Webcam Feed"
        />
        <canvas
          ref={overlayCanvasRef}
          className="overlay-canvas"
          aria-hidden="true"
        />

        {/* Live Status Badge overlay on video */}
        <div className="stream-badge-overlay">
          <span className={`badge-pill badge-${connectionStatus}`}>
            WS: {connectionStatus.toUpperCase()}
          </span>
          {isStreaming && (
            <span className="badge-pill badge-live">
              FEED ACTIVE
            </span>
          )}
        </div>
      </div>

      {/* Hidden canvas for frame rasterization */}
      <canvas ref={extractCanvasRef} style={{ display: 'none' }} />

      {cameraError && (
        <div className="camera-error-banner" role="alert">
          <AlertCircle size={18} />
          <span>{cameraError}</span>
        </div>
      )}

      <div className="controls">
        {!isStreaming ? (
          <button
            className="btn btn-primary"
            onClick={startCamera}
            id="btn-start-camera"
            disabled={cameraStatus === 'requesting'}
          >
            {cameraStatus === 'requesting' ? (
              <>
                <RefreshCw size={20} className="spinning" /> Initializing...
              </>
            ) : (
              <>
                <Video size={20} /> Start Monitoring
              </>
            )}
          </button>
        ) : (
          <button
            className="btn btn-danger"
            onClick={stopCamera}
            id="btn-stop-camera"
          >
            <VideoOff size={20} /> Stop Monitoring
          </button>
        )}
      </div>
    </div>
  );
};

export default CameraStream;
