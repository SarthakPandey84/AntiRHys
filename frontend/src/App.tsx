import { useState, useCallback } from 'react';
import { Activity, BarChart2 } from 'lucide-react';
import CameraStream from './components/CameraStream';
import Dashboard from './components/Dashboard';
import InsightsPage from './components/InsightsPage';
import type { DetectionState, ConnectionStatus, AppView } from './types';

function App() {
  const [activeView, setActiveView] = useState<AppView>('monitor');
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

  const handleRegisterSendCommand = useCallback(
    (fn: ((command: string, extra?: Record<string, unknown>) => boolean) | null) => {
      setSendCommandFn(() => fn);
    },
    []
  );

  return (
    <div className="app-container">
      <header className="header">
        <h1>AntiRHys</h1>
        <p>Driver Drowsiness & Road Hypnosis Monitoring Prototype</p>
        <nav className="app-nav-tabs" aria-label="Main Navigation">
          <button
            type="button"
            className={`app-nav-tab ${activeView === 'monitor' ? 'active' : ''}`}
            onClick={() => setActiveView('monitor')}
          >
            <Activity size={16} /> Live Monitor
          </button>
          <button
            type="button"
            className={`app-nav-tab ${activeView === 'insights' ? 'active' : ''}`}
            onClick={() => setActiveView('insights')}
          >
            <BarChart2 size={16} /> Insights & History
          </button>
        </nav>
      </header>

      <main className="main-content" style={{ display: activeView === 'monitor' ? 'flex' : 'none' }}>
        <CameraStream
          onDetectionUpdate={setDetectionState}
          onConnectionChange={setConnectionStatus}
          onRegisterSendCommand={handleRegisterSendCommand}
          landmarks={detectionState.landmarks}
        />
        <Dashboard
          data={detectionState}
          connectionStatus={connectionStatus}
          onSendCommand={sendCommandFn || undefined}
        />
      </main>

      {activeView === 'insights' && (
        <InsightsPage onSwitchToMonitor={() => setActiveView('monitor')} />
      )}
    </div>
  );
}

export default App;
