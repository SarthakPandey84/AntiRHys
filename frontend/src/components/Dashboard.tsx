import React from 'react';
import { Eye, EyeOff, Activity, AlertTriangle } from 'lucide-react';
import type { DetectionState } from '../App';

interface DashboardProps {
  data: DetectionState;
}

const Dashboard: React.FC<DashboardProps> = ({ data }) => {
  const isDrowsy = data.state === 'Drowsy';

  return (
    <div className="dashboard-container glass-panel">
      <div className={`status-card ${isDrowsy ? 'drowsy' : 'alert'}`}>
        {isDrowsy ? (
          <EyeOff size={64} color="#ef4444" />
        ) : (
          <Eye size={64} color="#22c55e" />
        )}
        <h2 style={{ marginTop: '1rem', fontSize: '2rem' }}>
          {data.state.toUpperCase()}
        </h2>
        {isDrowsy && (
          <p style={{ color: '#ef4444', marginTop: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertTriangle size={18} /> Driver is sleeping!
          </p>
        )}
      </div>

      <div className="stats-panel" style={{ marginTop: '1rem' }}>
        <div className="stat-row">
          <div className="stat-label">
            <Activity size={18} />
            EAR Value
          </div>
          <div className="stat-value" style={{ color: data.ear < 0.25 ? '#ef4444' : '#f8fafc' }}>
            {data.ear > 0 ? data.ear.toFixed(3) : '---'}
          </div>
        </div>

        <div className="stat-row">
          <div className="stat-label">
            <Eye size={18} />
            Blinks Count
          </div>
          <div className="stat-value">
            {data.blink_count}
          </div>
        </div>
        
        <div className="stat-row">
          <div className="stat-label">
            <AlertTriangle size={18} />
            EAR Threshold
          </div>
          <div className="stat-value" style={{ color: '#94a3b8', fontSize: '1.2rem' }}>
            0.25
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
