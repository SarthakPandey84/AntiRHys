import { useState } from 'react';
import CameraStream from './components/CameraStream';
import Dashboard from './components/Dashboard';
import type { DetectionState, ConnectionStatus } from './types';

function App() {
  const [detectionState, setDetectionState] = useState<DetectionState>({
    ear: 0,
    state: 'IDLE',
    detection_state: 'IDLE',
    tracking_state: 'LOST',
    eye_state: 'UNKNOWN',
    blink_count: 0,
    landmarks: [],
    perclos: 0,
    perclos_valid: false,
    perclos_status: 'CALCULATING',
    perclos_window_duration: 60,
    perclos_warning: false,
    calibration_state: 'IDLE',
    calibration_progress: 0.0,
    baseline_active: false,
    pitch: 0.0,
    yaw: 0.0,
    roll: 0.0,
    head_pose_valid: false,
    head_pose_status: 'UNAVAILABLE',
  });

  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [sendCommandFn, setSendCommandFn] = useState<((command: string, extra?: Record<string, unknown>) => boolean) | null>(null);

  return (
    <div className="app-container">
      <header className="header">
        <h1>AntiRHys</h1>
        <p>Driver Drowsiness & Road Hypnosis Monitoring Prototype</p>
      </header>

      <main className="main-content">
        <CameraStream
          onDetectionUpdate={setDetectionState}
          onConnectionChange={setConnectionStatus}
          onRegisterSendCommand={setSendCommandFn}
          landmarks={detectionState.landmarks}
        />
        <Dashboard
          data={detectionState}
          connectionStatus={connectionStatus}
          onSendCommand={sendCommandFn || undefined}
        />
      </main>
    </div>
  );
}

export default App;
