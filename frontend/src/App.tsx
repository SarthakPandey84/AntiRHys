import React, { useState } from 'react';
import CameraStream from './components/CameraStream';
import Dashboard from './components/Dashboard';

export interface DetectionState {
  ear: number;
  state: string;
  blink_count: number;
  landmarks: { x: number; y: number }[];
}

function App() {
  const [detectionState, setDetectionState] = useState<DetectionState>({
    ear: 0,
    state: 'Alert',
    blink_count: 0,
    landmarks: []
  });

  return (
    <div className="app-container">
      <header className="header">
        <h1>Driver Guardian</h1>
        <p>Real-time fatigue & road hypnosis detection</p>
      </header>

      <main className="main-content">
        <CameraStream 
          onDetectionUpdate={setDetectionState} 
          landmarks={detectionState.landmarks} 
        />
        <Dashboard data={detectionState} />
      </main>
    </div>
  );
}

export default App;
