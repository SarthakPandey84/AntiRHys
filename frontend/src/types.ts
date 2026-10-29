/**
 * AntiRHys Shared Frontend Types
 */

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

export type CameraStatus = 'idle' | 'requesting' | 'active' | 'denied' | 'error' | 'stopped';

export type CalibrationState = 'IDLE' | 'COLLECTING' | 'COMPLETED' | 'FAILED';

export type CalibrationFailureReason =
  | 'NONE'
  | 'INSUFFICIENT_SAMPLES'
  | 'INSUFFICIENT_DURATION'
  | 'LOW_TRACKING_RATIO'
  | 'SUSPECTED_DROWSINESS'
  | 'EXTREME_EAR_OUTLIER'
  | 'USER_ABORTED'
  | 'CONNECTION_DROPPED';

export type SystemDetectionState =
  | 'IDLE'
  | 'INITIALIZING'
  | 'CONNECTING'
  | 'NORMAL'
  | 'POSSIBLE_DROWSINESS'
  | 'TRACKING_LOST'
  | 'STALE'
  | 'DISCONNECTED'
  | 'CAMERA_OR_PROCESSING_ERROR'
  | 'ERROR'
  | 'UNAVAILABLE';

export interface DetectionState {
  ear: number;
  left_ear?: number;
  right_ear?: number;
  state: string; // "Alert" | "Drowsy" | "NORMAL" | "POSSIBLE_DROWSINESS" | "TRACKING_LOST"
  detection_state?: string;
  alarm_state?: 'OFF' | 'TRIGGERED';
  tracking_state?: 'ACTIVE' | 'LOST' | 'SEARCHING';
  eye_state?: 'OPEN' | 'CLOSED' | 'UNKNOWN';
  blink_count: number;
  closure_duration?: number;
  tracking_lost_duration?: number;
  session_id?: string;
  landmarks: { x: number; y: number }[];
  status_message?: string;
  last_received_time?: number;
  is_stale?: boolean;
  perclos?: number;
  perclos_valid?: boolean;
  perclos_status?: 'VALID' | 'CALCULATING' | 'INSUFFICIENT_DATA' | 'UNAVAILABLE' | string;
  perclos_window_duration?: number;
  perclos_warning?: boolean;
  baseline_active?: boolean;
  calibrated_ear_threshold?: number;
  calibration_state?: 'IDLE' | 'COLLECTING' | 'COMPLETED' | 'FAILED' | string;
  calibration_progress?: number;
  calibration_failure_reason?: string;
  pitch?: number;
  yaw?: number;
  roll?: number;
  head_pose_valid?: boolean;
  head_pose_status?: 'VALID' | 'FACE_LOST' | 'UNAVAILABLE' | 'DEGENERATE_GEOMETRY' | 'REPROJECTION_ERROR_HIGH' | string;
  head_pose_reprojection_error?: number;
  head_pose_confidence?: number;
  command?: string;
  client_session_id?: string;
  status?: string;
  error?: string;
}

export type AppView = 'monitor' | 'insights';

export type InsightsSubView = 'live' | 'history' | 'profile';

export type EventSeverity = 'INFO' | 'WARNING' | 'CRITICAL' | 'ERROR';

export interface SessionSummary {
  session_id: string;
  start_time: string;
  end_time: string;
  duration_sec: number;
  total_events: number;
  alert_count: number;
  error_count: number;
  mean_ear: number;
  mean_perclos: number;
  tracking_coverage_ratio: number;
  quality_status: 'GOOD' | 'FAIR' | 'INSUFFICIENT_TRACKING' | string;
  calibrated: boolean;
  calibrated_threshold: number | null;
  has_incomplete_data?: boolean;
}

export interface LogEvent {
  timestamp: string;
  timestamp_iso?: string;
  session_id: string;
  event_type: string;
  severity: EventSeverity;
  detection_state?: string;
  eye_state?: string;
  tracking_state?: string;
  ear?: number | null;
  blink_count?: number | null;
  closure_duration?: number | null;
  perclos?: number | null;
  perclos_status?: string;
  ear_threshold?: number | null;
  calibration_state?: string;
  detector_version?: string;
  notes?: string;
}

export interface SessionDetail {
  session_id: string;
  summary: SessionSummary;
  events: LogEvent[];
}

export interface PersonalizationProfileData {
  profile_version: number;
  driver_id: string;
  is_active: boolean;
  baseline_open_ear: number | null;
  personalized_threshold: number;
  min_threshold_bound: number;
  max_threshold_bound: number;
  sample_count: number;
  duration_sec: number;
  tracking_ratio: number;
  calibration_timestamp: number | null;
  quality_status: string;
  detector_version: string;
  update_reason: string;
  created_at?: number;
  updated_at?: number;
}
