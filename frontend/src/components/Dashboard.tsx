import React, { useEffect, useState, useRef } from 'react';
import {
  Eye,
  EyeOff,
  Activity,
  AlertTriangle,
  FileText,
  Volume2,
  VolumeX,
  BellRing,
  HelpCircle,
  Clock,
  ShieldAlert,
  WifiOff,
  Percent,
  Sliders,
  Sparkles,
  RotateCcw,
  XCircle,
  Play,
  Compass,
} from 'lucide-react';
import type { DetectionState, ConnectionStatus } from '../types';
import { audioAlertService, type AudioStatus } from '../utils/audioAlert';

interface DashboardProps {
  data: DetectionState;
  connectionStatus?: ConnectionStatus;
  onSendCommand?: (command: string, extra?: Record<string, unknown>) => boolean;
}

const Dashboard: React.FC<DashboardProps> = ({
  data,
  connectionStatus = 'disconnected',
  onSendCommand
}) => {
  const [audioStatus, setAudioStatus] = useState<AudioStatus>(audioAlertService.getStatus());
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [testPlaying, setTestPlaying] = useState<boolean>(false);
  const [commandPending, setCommandPending] = useState<boolean>(false);
  const commandTimeoutRef = useRef<number | null>(null);

  // Subscribe to audio status updates and cleanup timers
  useEffect(() => {
    const unsubscribe = audioAlertService.subscribe((status) => {
      setAudioStatus(status);
    });
    return () => {
      unsubscribe();
      audioAlertService.stopAlarm();
      if (commandTimeoutRef.current !== null) {
        clearTimeout(commandTimeoutRef.current);
      }
    };
  }, []);

  const isError =
    data.detection_state === 'CAMERA_OR_PROCESSING_ERROR' ||
    data.detection_state === 'ERROR' ||
    data.state === 'CAMERA_OR_PROCESSING_ERROR' ||
    data.state === 'ERROR' ||
    Boolean(data.error);

  const isInitializing =
    data.detection_state === 'INITIALIZING' ||
    data.state === 'INITIALIZING';

  const isDrowsy =
    !isError && (
      data.detection_state === 'POSSIBLE_DROWSINESS' ||
      data.alarm_state === 'TRIGGERED' ||
      data.state === 'Drowsy'
    );

  const isTrackingLost =
    !isError && (
      data.detection_state === 'TRACKING_LOST' ||
      data.tracking_state === 'LOST' ||
      data.state === 'TRACKING_LOST'
    );

  const isStale = data.is_stale || data.state === 'STALE';
  const isConnecting =
    connectionStatus === 'connecting' ||
    connectionStatus === 'reconnecting' ||
    data.state === 'CONNECTING';

  const isIdle =
    data.state === 'IDLE' ||
    (!data.session_id && connectionStatus === 'disconnected');

  // Trigger or halt audio alarm based on detection state
  useEffect(() => {
    if (isDrowsy) {
      audioAlertService.startAlarm();
    } else {
      audioAlertService.stopAlarm();
    }
  }, [isDrowsy]);

  const handleTestAudio = async () => {
    setTestPlaying(true);
    await audioAlertService.playTestSound();
    setTestPlaying(false);
  };

  const handleToggleMute = () => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    audioAlertService.setMuted(nextMuted);
  };

  // Calibration State derivations
  const calibState = data.calibration_state || 'IDLE';
  const isCollecting = calibState === 'COLLECTING';
  const isCalibCompleted = (calibState === 'COMPLETED' || calibState === 'IDLE') && Boolean(data.baseline_active);
  const isCalibFailed = calibState === 'FAILED';
  const progressPercent = Math.min(100, Math.max(0, Math.round((data.calibration_progress || 0) * 100)));
  const isCommandActive = commandPending && connectionStatus === 'connected';

  const handleStartCalibration = () => {
    if (isCommandActive || !onSendCommand) return;
    setCommandPending(true);
    onSendCommand('start_calibration', { driver_id: 'driver' });
    if (commandTimeoutRef.current !== null) clearTimeout(commandTimeoutRef.current);
    commandTimeoutRef.current = window.setTimeout(() => setCommandPending(false), 800);
  };

  const handleAbortCalibration = () => {
    if (isCommandActive || !onSendCommand) return;
    setCommandPending(true);
    onSendCommand('abort_calibration');
    if (commandTimeoutRef.current !== null) clearTimeout(commandTimeoutRef.current);
    commandTimeoutRef.current = window.setTimeout(() => setCommandPending(false), 800);
  };

  const handleClearBaseline = () => {
    if (isCommandActive || !onSendCommand) return;
    setCommandPending(true);
    onSendCommand('clear_baseline');
    if (commandTimeoutRef.current !== null) clearTimeout(commandTimeoutRef.current);
    commandTimeoutRef.current = window.setTimeout(() => setCommandPending(false), 800);
  };

  const getFailureExplanation = (reason?: string) => {
    switch (reason) {
      case 'USER_ABORTED':
        return 'Calibration was aborted by user. Using standard default threshold (0.250).';
      case 'SUSPECTED_DROWSINESS':
        return 'Calibration rejected: excessive eye closure or drowsiness detected during window. Keep eyes open and stay attentive.';
      case 'LOW_TRACKING_RATIO':
        return 'Calibration rejected: face tracking stability was below 80%. Ensure face is centered and adequately lit.';
      case 'INSUFFICIENT_SAMPLES':
      case 'INSUFFICIENT_DURATION':
        return 'Calibration incomplete: insufficient valid face observations collected.';
      case 'EXTREME_EAR_OUTLIER':
        return 'Calibration rejected: ocular measurements outside plausible physiological limits (0.20 – 0.50).';
      default:
        return 'Calibration rejected: quality criteria not met. Using standard default threshold (0.250).';
    }
  };

  // Effective EAR threshold (personalized if active, otherwise default 0.25)
  const effectiveThreshold = data.baseline_active && data.calibrated_ear_threshold !== undefined
    ? data.calibrated_ear_threshold
    : 0.25;

  // Derive visual card theme and title
  let cardClass = 'status-card';
  let statusTitle: string;
  let statusDescription: string;
  let StatusIcon: React.ElementType;
  let iconColor: string;

  if (isIdle) {
    cardClass += ' idle';
    statusTitle = 'SESSION IDLE';
    statusDescription = 'Click "Start Monitoring" to initiate webcam tracking.';
    StatusIcon = Eye;
    iconColor = '#94a3b8';
  } else if (connectionStatus === 'error' || connectionStatus === 'disconnected') {
    cardClass += ' disconnected';
    statusTitle = 'BACKEND DISCONNECTED';
    statusDescription = 'Cannot connect to detection service. Ensure backend is running.';
    StatusIcon = WifiOff;
    iconColor = '#ef4444';
  } else if (isError) {
    cardClass += ' error';
    statusTitle = 'PROCESSING ERROR';
    statusDescription = data.error || data.status_message || 'Image processing error encountered.';
    StatusIcon = ShieldAlert;
    iconColor = '#ef4444';
  } else if (isInitializing) {
    cardClass += ' initializing';
    statusTitle = 'INITIALIZING...';
    statusDescription = data.status_message || 'Initializing facial landmark detector.';
    StatusIcon = Activity;
    iconColor = '#60a5fa';
  } else if (isConnecting) {
    cardClass += ' connecting';
    statusTitle = 'CONNECTING...';
    statusDescription = connectionStatus === 'reconnecting' ? 'Connecting to detection server... (Cloud backend may take ~30-50s to wake up if idle).' : 'Establishing real-time connection with detection server.';
    StatusIcon = Activity;
    iconColor = '#eab308';
  } else if (isStale) {
    cardClass += ' stale';
    statusTitle = 'DETECTION STALE';
    statusDescription = 'Backend response delayed. Frame processing is lagging.';
    StatusIcon = Clock;
    iconColor = '#f97316';
  } else if (isTrackingLost) {
    if (isDrowsy) {
      cardClass += ' tracking-lost drowsy';
      statusTitle = 'TRACKING LOST (ALARM ACTIVE)';
      statusDescription =
        'Face tracking lost while drowsiness alarm is active! Keep face centered and stay alert.';
      StatusIcon = AlertTriangle;
      iconColor = '#ef4444';
    } else {
      cardClass += ' tracking-lost';
      statusTitle = 'TRACKING LOST';
      statusDescription =
        data.status_message || 'Face not detected in frame. Please keep your face centered.';
      StatusIcon = AlertTriangle;
      iconColor = '#f59e0b';
    }
  } else if (isDrowsy) {
    cardClass += ' drowsy';
    statusTitle = 'POSSIBLE DROWSINESS';
    statusDescription = data.status_message || 'Prolonged eye closure detected. Please take a break!';
    StatusIcon = EyeOff;
    iconColor = '#ef4444';
  } else {
    cardClass += ' alert';
    statusTitle = 'MONITORING NORMALLY';
    statusDescription = data.status_message || 'Eyes open. Tracking active.';
    StatusIcon = Eye;
    iconColor = '#22c55e';
  }

  const isLoggingActive =
    Boolean(data.session_id) &&
    connectionStatus === 'connected' &&
    !isIdle &&
    data.state !== 'UNAVAILABLE';

  return (
    <div className="dashboard-container glass-panel">
      {/* Audio Readiness Banner */}
      <div className="audio-control-card">
        <div className="audio-info">
          <div className="audio-label">
            {isMuted ? (
              <VolumeX size={18} color="#ef4444" />
            ) : (
              <Volume2 size={18} color={audioStatus === 'ready' ? '#22c55e' : '#eab308'} />
            )}
            <span>
              Audio Alert:{' '}
              <strong>
                {isMuted ? 'Muted' : audioStatus === 'ready' ? 'Ready' : 'Not Tested'}
              </strong>
            </span>
          </div>
          {audioStatus !== 'ready' && !isMuted && (
            <p className="audio-hint">
              Test sound before driving to clear browser autoplay restrictions.
            </p>
          )}
        </div>

        <div className="audio-actions">
          <button
            className="btn btn-secondary btn-sm"
            onClick={handleTestAudio}
            disabled={testPlaying || isMuted}
            id="btn-test-audio"
            title="Play a safe sound test to verify browser audio"
          >
            <BellRing size={16} />
            {testPlaying ? 'Testing...' : 'Test Sound'}
          </button>
          <button
            className={`btn ${isMuted ? 'btn-danger' : 'btn-secondary'} btn-sm`}
            onClick={handleToggleMute}
            id="btn-toggle-mute"
            title={isMuted ? 'Unmute alarm' : 'Mute alarm'}
          >
            {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
            {isMuted ? 'Unmute' : 'Mute'}
          </button>
        </div>
      </div>

      {isMuted && (
        <div className="safety-warning-banner" role="alert">
          <ShieldAlert size={18} />
          <span>Warning: Audible alarm is muted. Audible warnings will not sound.</span>
        </div>
      )}

      {/* Main Status Display */}
      <div className={cardClass} role="status" aria-live="polite">
        <StatusIcon size={60} color={iconColor} />
        <h2 className="status-title">{statusTitle}</h2>
        <p className="status-desc">{statusDescription}</p>
      </div>

      {/* Personalized Driver Calibration Card */}
      <div
        className={`calibration-card ${
          isCollecting ? 'collecting' : isCalibCompleted ? 'completed' : isCalibFailed ? 'failed' : ''
        }`}
        role="region"
        aria-label="Personalized Driver Calibration"
      >
        <div className="calibration-header">
          <div className="calibration-title-wrap">
            <Sliders size={18} color={isCalibCompleted ? '#22c55e' : isCollecting ? '#60a5fa' : '#94a3b8'} />
            <span>Driver Baseline Calibration</span>
          </div>
          <span
            className={`calibration-badge ${
              isCollecting
                ? 'collecting'
                : isCalibCompleted
                ? 'completed'
                : isCalibFailed
                ? 'failed'
                : 'idle'
            }`}
          >
            {isCollecting
              ? `COLLECTING (${progressPercent}%)`
              : isCalibCompleted
              ? 'BASELINE ACTIVE'
              : isCalibFailed
              ? data.calibration_failure_reason === 'USER_ABORTED'
                ? 'ABORTED'
                : 'REJECTED'
              : 'IDLE'}
          </span>
        </div>

        {/* Dynamic Status & Guidance Content */}
        {isCollecting ? (
          <>
            <p className="calibration-description" style={{ color: '#93c5fd' }}>
              Measuring open-eye EAR aperture... Look straight ahead, maintain a natural gaze, and keep eyes open.
            </p>
            <div
              className="calibration-progress-track"
              role="progressbar"
              aria-valuenow={progressPercent}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Calibration Progress"
            >
              <div
                className="calibration-progress-fill"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
            <div className="calibration-actions">
              <button
                className="btn btn-secondary btn-sm"
                onClick={handleAbortCalibration}
                disabled={isCommandActive || connectionStatus !== 'connected'}
                id="btn-abort-calibration"
              >
                <XCircle size={16} /> Cancel Calibration
              </button>
            </div>
          </>
        ) : isCalibCompleted ? (
          <>
            <p className="calibration-description" style={{ color: '#86efac' }}>
              Personalized baseline active ({data.calibrated_ear_threshold?.toFixed(3)} EAR closure threshold). Detection state machine adapted to your natural eye shape.
            </p>
            <div className="calibration-actions">
              <button
                className="btn btn-secondary btn-sm"
                onClick={handleStartCalibration}
                disabled={isCommandActive || isIdle || connectionStatus !== 'connected'}
                id="btn-recalibrate"
                title="Recalibrate open-eye baseline"
              >
                <RotateCcw size={16} /> Recalibrate
              </button>
              <button
                className="btn btn-secondary btn-sm"
                onClick={handleClearBaseline}
                disabled={isCommandActive || connectionStatus !== 'connected'}
                id="btn-clear-baseline"
                title="Restore default threshold (0.250)"
              >
                <XCircle size={16} /> Clear Baseline
              </button>
            </div>
          </>
        ) : isCalibFailed ? (
          <>
            <p className="calibration-description" style={{ color: '#fca5a5' }}>
              {getFailureExplanation(data.calibration_failure_reason)}
            </p>
            <div className="calibration-actions">
              <button
                className="btn btn-primary btn-sm"
                onClick={handleStartCalibration}
                disabled={isCommandActive || isIdle || connectionStatus !== 'connected'}
                id="btn-retry-calibration"
              >
                <RotateCcw size={16} /> Try Calibration Again
              </button>
              {data.baseline_active && (
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={handleClearBaseline}
                  disabled={isCommandActive || connectionStatus !== 'connected'}
                  id="btn-clear-baseline"
                >
                  <XCircle size={16} /> Clear Baseline
                </button>
              )}
            </div>
          </>
        ) : (
          <>
            <p className="calibration-description">
              Calibrate your individual open-eye baseline over a 12-second window to reduce false alarms from naturally narrow or hooded eyes.
            </p>
            <div className="calibration-actions">
              <button
                className="btn btn-primary btn-sm"
                onClick={handleStartCalibration}
                disabled={isCommandActive || isIdle || connectionStatus !== 'connected'}
                id="btn-start-calibration"
                title={
                  isIdle
                    ? 'Start webcam monitoring first to enable calibration'
                    : connectionStatus !== 'connected'
                    ? 'Backend must be connected to calibrate'
                    : 'Start a 12-second attentive baseline calibration'
                }
              >
                <Play size={16} /> Start Calibration
              </button>
              {isIdle && (
                <span className="audio-hint" style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                  (Start camera monitoring first)
                </span>
              )}
            </div>
          </>
        )}

        {/* Ethical & Safety Transparency Disclaimer */}
        <div className="calibration-disclaimer">
          <strong>Notice:</strong> Baseline calibration tunes the eye-closure threshold to individual ocular anatomy. It does <em>not</em> certify that you are alert, rested, or safe to operate a vehicle.
        </div>
      </div>

      {/* Numerical Metrics Panel */}
      <div className="stats-panel">
        <div className="stat-row">
          <div className="stat-label">
            <Activity size={18} />
            <span>Eye Aspect Ratio (EAR)</span>
          </div>
          <div
            className="stat-value"
            style={{
              color:
                data.ear > 0 && data.ear < effectiveThreshold ? '#ef4444' : '#f8fafc',
            }}
          >
            {data.ear > 0 ? data.ear.toFixed(3) : '---'}
          </div>
        </div>

        {data.closure_duration !== undefined && data.closure_duration > 0 && (
          <div className="stat-row">
            <div className="stat-label">
              <Clock size={18} />
              <span>Current Closure Duration</span>
            </div>
            <div className="stat-value" style={{ color: '#ef4444' }}>
              {data.closure_duration.toFixed(1)}s
            </div>
          </div>
        )}

        <div className="stat-row">
          <div className="stat-label">
            <Eye size={18} />
            <span>Blinks Detected</span>
          </div>
          <div className="stat-value">{data.blink_count ?? 0}</div>
        </div>

        <div className="stat-row">
          <div className="stat-label">
            <Percent size={18} />
            <span>PERCLOS (60s Window)</span>
          </div>
          <div className="stat-value" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span
              style={{
                color:
                  data.perclos_warning
                    ? '#ef4444'
                    : data.perclos_valid
                    ? '#f8fafc'
                    : '#94a3b8',
              }}
            >
              {data.perclos !== undefined && data.perclos_valid
                ? `${(data.perclos * 100).toFixed(1)}%`
                : '---'}
            </span>
            <span
              className={`badge-pill badge-status-${(data.perclos_status || 'CALCULATING').toLowerCase()}`}
              style={{
                fontSize: '0.65rem',
                padding: '0.2rem 0.5rem',
                borderRadius: '4px',
                background:
                  data.perclos_status === 'VALID'
                    ? 'rgba(34, 197, 94, 0.2)'
                    : 'rgba(234, 179, 8, 0.2)',
                color:
                  data.perclos_status === 'VALID' ? '#22c55e' : '#eab308',
                border:
                  data.perclos_status === 'VALID'
                    ? '1px solid rgba(34, 197, 94, 0.4)'
                    : '1px solid rgba(234, 179, 8, 0.4)',
              }}
            >
              {data.perclos_status || 'CALCULATING'}
            </span>
          </div>
        </div>

        <div className="stat-row">
          <div className="stat-label">
            {data.baseline_active ? (
              <Sparkles size={18} color="#22c55e" />
            ) : (
              <AlertTriangle size={18} />
            )}
            <span>
              {data.baseline_active ? 'Personalized EAR Threshold' : 'Default EAR Threshold'}
            </span>
          </div>
          <div
            className="stat-value"
            style={{
              color: data.baseline_active ? '#22c55e' : '#f8fafc',
            }}
          >
            {data.baseline_active && data.calibrated_ear_threshold !== undefined
              ? data.calibrated_ear_threshold.toFixed(3)
              : '0.250'}{' '}
            <span
              className="threshold-sub"
              style={{
                color: data.baseline_active ? '#86efac' : undefined,
              }}
            >
              {data.baseline_active ? '(Active)' : '(≥ 1.5s)'}
            </span>
          </div>
        </div>

        <div className="stat-row">
          <div className="stat-label">
            <Compass size={18} />
            <span>Head Pose (Pitch / Yaw / Roll)</span>
          </div>
          <div className="stat-value" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}>
            {data.head_pose_valid ? (
              <>
                <span title="Pitch: negative = nod down, positive = up">
                  P: {data.pitch !== undefined ? `${data.pitch > 0 ? '+' : ''}${data.pitch.toFixed(1)}°` : '0.0°'}
                </span>
                <span style={{ color: '#64748b' }}>|</span>
                <span title="Yaw: negative = right, positive = left">
                  Y: {data.yaw !== undefined ? `${data.yaw > 0 ? '+' : ''}${data.yaw.toFixed(1)}°` : '0.0°'}
                </span>
                <span style={{ color: '#64748b' }}>|</span>
                <span title="Roll: negative = tilt left, positive = tilt right">
                  R: {data.roll !== undefined ? `${data.roll > 0 ? '+' : ''}${data.roll.toFixed(1)}°` : '0.0°'}
                </span>
              </>
            ) : (
              <span style={{ color: '#94a3b8' }}>
                {data.head_pose_status === 'FACE_LOST' ? 'Face Lost' : '---'}
              </span>
            )}
            <span
              className="badge-pill"
              style={{
                fontSize: '0.65rem',
                padding: '0.15rem 0.4rem',
                borderRadius: '4px',
                background: data.head_pose_valid ? 'rgba(34, 197, 94, 0.2)' : 'rgba(148, 163, 184, 0.15)',
                color: data.head_pose_valid ? '#22c55e' : '#94a3b8',
                border: data.head_pose_valid ? '1px solid rgba(34, 197, 94, 0.4)' : '1px solid rgba(148, 163, 184, 0.3)',
              }}
            >
              {data.head_pose_valid ? '3D PnP' : 'RESEARCH'}
            </span>
          </div>
        </div>

        <div className="stat-row session-logging-row">
          <div className="stat-label">
            <FileText size={18} />
            <span>Session Logging</span>
          </div>
          <div
            className="stat-value"
            style={{
              fontSize: '0.85rem',
              color: isLoggingActive ? '#22c55e' : '#94a3b8',
            }}
          >
            {isLoggingActive ? 'Active (session_log.csv)' : 'Inactive'}
          </div>
        </div>

        {/* Prototype Transparency Notice */}
        <div className="prototype-notice">
          <HelpCircle size={14} />
          <span>
            AntiRHys is a research prototype. EAR alone does not guarantee detection of all sleep,
            fatigue, or road hypnosis events.
          </span>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
