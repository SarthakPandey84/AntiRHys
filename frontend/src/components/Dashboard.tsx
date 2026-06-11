import React, { useEffect, useRef, useState } from 'react';
import { Eye, EyeOff, Activity, AlertTriangle, FileText } from 'lucide-react';
import type { DetectionState } from '../App';

interface DashboardProps {
  data: DetectionState;
}

// Create a single AudioContext to avoid hitting browser limits
let audioCtx: AudioContext | null = null;

const initAudio = () => {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
};

let isHighTone = false;
const playAlarm = () => {
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator();
  const gainNode = audioCtx.createGain();
  
  // High volume sine wave for siren
  osc.type = 'sine';
  const freq = isHighTone ? 1000 : 800;
  isHighTone = !isHighTone;
  
  osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
  
  osc.connect(gainNode);
  gainNode.connect(audioCtx.destination);
  
  // Set high gain
  gainNode.gain.setValueAtTime(0.8, audioCtx.currentTime);
  gainNode.gain.setValueAtTime(0.8, audioCtx.currentTime + 0.45);
  gainNode.gain.exponentialRampToValueAtTime(0.00001, audioCtx.currentTime + 0.5);
  
  osc.start();
  osc.stop(audioCtx.currentTime + 0.5);
};

const Dashboard: React.FC<DashboardProps> = ({ data }) => {
  const isDrowsy = data.state === 'Drowsy';
  const alarmIntervalRef = useRef<number | null>(null);
  const [alarmEnabled, setAlarmEnabled] = useState(false);

  useEffect(() => {
    // We try to init audio when component mounts or state changes
    // But user interaction is usually required first
    if (isDrowsy && alarmEnabled) {
      if (!alarmIntervalRef.current) {
        initAudio();
        isHighTone = false; // reset
        playAlarm(); // play immediately
        alarmIntervalRef.current = window.setInterval(playAlarm, 500);
      }
    } else {
      if (alarmIntervalRef.current) {
        clearInterval(alarmIntervalRef.current);
        alarmIntervalRef.current = null;
      }
    }
  }, [isDrowsy, alarmEnabled]);

  const handleEnableAlarm = () => {
    initAudio();
    setAlarmEnabled(true);
  };

  return (
    <div className="dashboard-container glass-panel">
      {!alarmEnabled && (
        <div style={{ marginBottom: '1rem', background: 'rgba(255,200,0,0.1)', padding: '0.5rem', borderRadius: '8px', border: '1px solid #eab308' }}>
          <p style={{ color: '#eab308', fontSize: '0.9rem', marginBottom: '0.5rem' }}>
            Audible alarm is disabled. Click below to enable audio.
          </p>
          <button className="btn btn-primary" onClick={handleEnableAlarm} style={{ padding: '0.4rem 0.8rem', fontSize: '0.9rem' }}>
            Enable Alarm
          </button>
        </div>
      )}

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

        <div className="stat-row" style={{ marginTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '1rem' }}>
          <div className="stat-label">
            <FileText size={18} />
            Data Logging
          </div>
          <div className="stat-value" style={{ fontSize: '0.9rem', color: '#22c55e' }}>
            Active (session_log.csv)
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
