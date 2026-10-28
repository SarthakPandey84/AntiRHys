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
  | 'USER_ABORTED';

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
