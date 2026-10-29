import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Activity,
  History,
  Sliders,
  Download,
  Trash2,
  RefreshCw,
  Play,
  Pause,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Eye,
  ShieldCheck,
  ShieldAlert,
  ArrowLeft,
  Search,
  Filter,
  ExternalLink,
  ChevronRight
} from 'lucide-react';
import { getApiBaseUrl, getAuthToken } from '../config';
import type {
  InsightsSubView,
  SessionSummary,
  LogEvent,
  SessionDetail,
  PersonalizationProfileData,
  EventSeverity
} from '../types';

interface InsightsPageProps {
  onSwitchToMonitor?: () => void;
}

export default function InsightsPage({ onSwitchToMonitor }: InsightsPageProps) {
  const [subView, setSubView] = useState<InsightsSubView>('history');

  // Live events state
  const [liveEvents, setLiveEvents] = useState<LogEvent[]>([]);
  const [isLivePaused, setIsLivePaused] = useState(false);
  const [liveSeverityFilter, setLiveSeverityFilter] = useState<string>('ALL');
  const [liveSearchText, setLiveSearchText] = useState('');

  // Sessions state
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [sessionDetail, setSessionDetail] = useState<SessionDetail | null>(null);
  const [sessionSearchText, setSessionSearchText] = useState('');

  // Personalization profile state
  const [profile, setProfile] = useState<PersonalizationProfileData | null>(null);

  // Status & Feedback
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const apiBase = getApiBaseUrl();
  const authToken = getAuthToken();

  // Helper for authenticated fetch
  const authFetch = useCallback(
    async (endpoint: string, options: RequestInit = {}): Promise<Response> => {
      const headers = new Headers(options.headers || {});
      if (authToken) {
        headers.set('Authorization', `Bearer ${authToken}`);
      }
      const sep = endpoint.includes('?') ? '&' : '?';
      const url = authToken
        ? `${apiBase}${endpoint}${sep}token=${encodeURIComponent(authToken)}`
        : `${apiBase}${endpoint}`;

      return fetch(url, { ...options, headers });
    },
    [apiBase, authToken]
  );

  // Clear messages after 5 seconds
  useEffect(() => {
    if (successMessage) {
      const timer = setTimeout(() => setSuccessMessage(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [successMessage]);

  // Load Sessions
  const loadSessions = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await authFetch('/api/sessions');
      if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch sessions`);
      const data = await res.json();
      setSessions(data.sessions || []);
      setErrorMessage(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(`Error loading sessions: ${msg}`);
    } finally {
      setIsLoading(false);
    }
  }, [authFetch]);

  // Load Session Detail Timeline
  const loadSessionDetail = useCallback(
    async (sessionId: string) => {
      setIsLoading(true);
      try {
        const res = await authFetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch session timeline`);
        const data = await res.json();
        setSessionDetail(data);
        setSelectedSessionId(sessionId);
        setErrorMessage(null);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        setErrorMessage(`Error loading session timeline: ${msg}`);
      } finally {
        setIsLoading(false);
      }
    },
    [authFetch]
  );

  // Load Personalization Profile
  const loadProfile = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await authFetch('/api/profile');
      if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch profile`);
      const data = await res.json();
      setProfile(data);
      setErrorMessage(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(`Error loading profile: ${msg}`);
    } finally {
      setIsLoading(false);
    }
  }, [authFetch]);

  // Load Live Events
  const loadLiveEvents = useCallback(async () => {
    try {
      const res = await authFetch('/api/events/live?limit=100');
      if (res.ok) {
        const data = await res.json();
        setLiveEvents(data.events || []);
      }
    } catch {
      // Background poll failure suppressed to avoid flashing banners
    }
  }, [authFetch]);

  // Initial tab loading
  useEffect(() => {
    let ignore = false;
    const fetchTabData = async () => {
      try {
        if (subView === 'history') {
          setIsLoading(true);
          const res = await authFetch('/api/sessions');
          if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch sessions`);
          const data = await res.json();
          if (!ignore) {
            setSessions(data.sessions || []);
            setErrorMessage(null);
          }
        } else if (subView === 'profile') {
          setIsLoading(true);
          const res = await authFetch('/api/profile');
          if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch profile`);
          const data = await res.json();
          if (!ignore) {
            setProfile(data);
            setErrorMessage(null);
          }
        } else if (subView === 'live') {
          const res = await authFetch('/api/events/live?limit=100');
          if (res.ok) {
            const data = await res.json();
            if (!ignore) setLiveEvents(data.events || []);
          }
        }
      } catch (err: unknown) {
        if (!ignore) {
          const msg = err instanceof Error ? err.message : String(err);
          setErrorMessage(msg);
        }
      } finally {
        if (!ignore) setIsLoading(false);
      }
    };

    fetchTabData();
    return () => {
      ignore = true;
    };
  }, [subView, authFetch]);

  // Live polling interval
  useEffect(() => {
    if (subView !== 'live' || isLivePaused) return;

    const interval = setInterval(() => {
      loadLiveEvents();
    }, 2500);

    return () => clearInterval(interval);
  }, [subView, isLivePaused, loadLiveEvents]);

  // Export handlers
  const downloadBlob = (content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExportSession = async (sessionId: string, format: 'csv' | 'json') => {
    try {
      const res = await authFetch(`/api/sessions/${encodeURIComponent(sessionId)}/export?format=${format}`);
      if (!res.ok) throw new Error(`Export failed (HTTP ${res.status})`);
      const content = await res.text();
      const mime = format === 'csv' ? 'text/csv' : 'application/json';
      downloadBlob(content, `antirhys_${sessionId}.${format}`, mime);
      setSuccessMessage(`Session exported as ${format.toUpperCase()}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(`Export failed: ${msg}`);
    }
  };

  const handleExportAll = async (format: 'csv' | 'json') => {
    try {
      const res = await authFetch(`/api/export/all?format=${format}`);
      if (!res.ok) throw new Error(`Export all failed (HTTP ${res.status})`);
      const content = await res.text();
      const mime = format === 'csv' ? 'text/csv' : 'application/json';
      downloadBlob(content, `antirhys_all_sessions.${format}`, mime);
      setSuccessMessage(`All sessions exported as ${format.toUpperCase()}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(`Export all failed: ${msg}`);
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    if (!window.confirm(`Are you sure you want to delete session ${sessionId}? This cannot be undone.`)) {
      return;
    }
    try {
      const res = await authFetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`Delete failed (HTTP ${res.status})`);
      setSuccessMessage(`Session ${sessionId} deleted.`);
      if (selectedSessionId === sessionId) {
        setSelectedSessionId(null);
        setSessionDetail(null);
      }
      loadSessions();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(`Failed to delete session: ${msg}`);
    }
  };

  const handleClearAllSessions = async () => {
    if (!window.confirm('Are you sure you want to clear ALL logged sessions? This permanently deletes all historical records.')) {
      return;
    }
    try {
      const res = await authFetch('/api/sessions', { method: 'DELETE' });
      if (!res.ok) throw new Error(`Clear logs failed (HTTP ${res.status})`);
      setSuccessMessage('All session logs have been cleared.');
      setSelectedSessionId(null);
      setSessionDetail(null);
      loadSessions();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(`Failed to clear session logs: ${msg}`);
    }
  };

  const handleResetProfile = async () => {
    if (!window.confirm('Reset personalization profile to factory defaults (0.250)? This will clear your custom baseline.')) {
      return;
    }
    try {
      const res = await authFetch('/api/profile/reset', { method: 'POST' });
      if (!res.ok) throw new Error(`Reset failed (HTTP ${res.status})`);
      const data = await res.json();
      setProfile(data.profile);
      setSuccessMessage('Personalization profile reset to factory defaults.');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(`Failed to reset profile: ${msg}`);
    }
  };

  // Filtered live events
  const filteredLiveEvents = useMemo(() => {
    return liveEvents.filter((ev) => {
      const matchesSeverity = liveSeverityFilter === 'ALL' || ev.severity === liveSeverityFilter;
      const matchesSearch =
        !liveSearchText ||
        ev.event_type.toLowerCase().includes(liveSearchText.toLowerCase()) ||
        (ev.notes && ev.notes.toLowerCase().includes(liveSearchText.toLowerCase())) ||
        (ev.session_id && ev.session_id.toLowerCase().includes(liveSearchText.toLowerCase()));
      return matchesSeverity && matchesSearch;
    });
  }, [liveEvents, liveSeverityFilter, liveSearchText]);

  // Filtered sessions
  const filteredSessions = useMemo(() => {
    if (!sessionSearchText) return sessions;
    return sessions.filter((s) => s.session_id.toLowerCase().includes(sessionSearchText.toLowerCase()));
  }, [sessions, sessionSearchText]);

  // Helper formatting functions
  const formatDuration = (seconds: number) => {
    if (!seconds || seconds <= 0) return '< 1s';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    if (m === 0) return `${s}s`;
    return `${m}m ${s}s`;
  };

  const formatTimestamp = (iso?: string) => {
    if (!iso) return '—';
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return iso;
      return d.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return iso;
    }
  };

  const formatDateTime = (iso: string) => {
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return iso;
      return d.toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      });
    } catch {
      return iso;
    }
  };

  const renderSeverityBadge = (severity: EventSeverity) => {
    const sev = (severity || 'INFO').toUpperCase();
    let badgeClass = 'badge-info';
    if (sev === 'CRITICAL') badgeClass = 'badge-critical';
    else if (sev === 'WARNING') badgeClass = 'badge-warning';
    else if (sev === 'ERROR') badgeClass = 'badge-error';

    return <span className={`insights-severity-badge ${badgeClass}`}>{sev}</span>;
  };

  const renderQualityBadge = (status: string) => {
    const st = (status || 'UNAVAILABLE').toUpperCase();
    if (st === 'GOOD') {
      return <span className="insights-quality-badge quality-good"><CheckCircle2 size={13} /> Good</span>;
    }
    if (st === 'FAIR') {
      return <span className="insights-quality-badge quality-fair"><AlertTriangle size={13} /> Fair</span>;
    }
    return <span className="insights-quality-badge quality-poor"><ShieldAlert size={13} /> Insufficient Tracking</span>;
  };

  return (
    <div className="insights-container glass-panel">
      {/* Safety Notice Banner */}
      <div className="insights-safety-banner" role="alert">
        <AlertTriangle className="safety-banner-icon" size={20} />
        <div className="safety-banner-content">
          <strong>Safety Critical Notice:</strong> AntiRHys is an experimental research prototype for driver drowsiness
          monitoring. A driving session without logged alerts does <em>not</em> guarantee driver vigilance or road
          safety. Never drive when fatigued, regardless of monitoring output.
        </div>
      </div>

      {/* Sub-navigation Bar */}
      <div className="insights-nav-bar">
        <div className="insights-tabs">
          <button
            className={`insights-tab-btn ${subView === 'history' ? 'active' : ''}`}
            onClick={() => {
              setSubView('history');
              setSelectedSessionId(null);
              setSessionDetail(null);
            }}
          >
            <History size={16} />
            <span>Session History</span>
            <span className="insights-count-pill">{sessions.length}</span>
          </button>

          <button
            className={`insights-tab-btn ${subView === 'live' ? 'active' : ''}`}
            onClick={() => setSubView('live')}
          >
            <Activity size={16} />
            <span>Live Logs</span>
            <span className="insights-count-pill live-pill">{liveEvents.length}</span>
          </button>

          <button
            className={`insights-tab-btn ${subView === 'profile' ? 'active' : ''}`}
            onClick={() => setSubView('profile')}
          >
            <Sliders size={16} />
            <span>Personalization</span>
            {profile?.is_active ? (
              <span className="insights-count-pill active-profile-pill">Active</span>
            ) : (
              <span className="insights-count-pill default-profile-pill">Default</span>
            )}
          </button>
        </div>

        <div className="insights-nav-actions">
          {subView === 'history' && !selectedSessionId && (
            <>
              <button
                className="insights-action-btn"
                onClick={() => handleExportAll('csv')}
                title="Export all session logs as CSV"
              >
                <Download size={14} /> Export All (CSV)
              </button>
              <button
                className="insights-action-btn"
                onClick={() => handleExportAll('json')}
                title="Export all session logs as JSON"
              >
                <Download size={14} /> Export All (JSON)
              </button>
              <button
                className="insights-action-btn danger-btn"
                onClick={handleClearAllSessions}
                title="Delete all recorded session logs"
              >
                <Trash2 size={14} /> Clear All
              </button>
              <button className="insights-action-btn" onClick={loadSessions} title="Refresh session list">
                <RefreshCw size={14} className={isLoading ? 'spin-icon' : ''} />
              </button>
            </>
          )}

          {subView === 'live' && (
            <>
              <button
                className={`insights-action-btn ${isLivePaused ? 'active' : ''}`}
                onClick={() => setIsLivePaused((p) => !p)}
              >
                {isLivePaused ? <Play size={14} /> : <Pause size={14} />}
                {isLivePaused ? 'Resume Stream' : 'Pause Stream'}
              </button>
              <button className="insights-action-btn" onClick={loadLiveEvents} title="Refresh live buffer">
                <RefreshCw size={14} className={isLoading ? 'spin-icon' : ''} />
              </button>
            </>
          )}

          {subView === 'profile' && (
            <button className="insights-action-btn" onClick={loadProfile} title="Reload profile">
              <RefreshCw size={14} className={isLoading ? 'spin-icon' : ''} />
            </button>
          )}
        </div>
      </div>

      {/* Notifications */}
      {errorMessage && (
        <div className="insights-message-bar error-bar">
          <AlertTriangle size={16} />
          <span>{errorMessage}</span>
          <button className="close-btn" onClick={() => setErrorMessage(null)}>
            ×
          </button>
        </div>
      )}
      {successMessage && (
        <div className="insights-message-bar success-bar">
          <CheckCircle2 size={16} />
          <span>{successMessage}</span>
          <button className="close-btn" onClick={() => setSuccessMessage(null)}>
            ×
          </button>
        </div>
      )}

      {/* VIEW 1: SESSION HISTORY (LIST & DETAIL) */}
      {subView === 'history' && (
        <div className="insights-view-content">
          {selectedSessionId && sessionDetail ? (
            /* Session Detail View */
            <div className="session-detail-view">
              <div className="detail-header">
                <button
                  className="back-btn"
                  onClick={() => {
                    setSelectedSessionId(null);
                    setSessionDetail(null);
                  }}
                >
                  <ArrowLeft size={16} /> Back to Sessions
                </button>
                <div className="detail-title-group">
                  <h3>Session {sessionDetail.session_id}</h3>
                  <div className="detail-tags">
                    {renderQualityBadge(sessionDetail.summary.quality_status)}
                    {sessionDetail.summary.calibrated ? (
                      <span className="calibrated-tag">
                        Calibrated ({sessionDetail.summary.calibrated_threshold?.toFixed(3) || '0.250'})
                      </span>
                    ) : (
                      <span className="uncalibrated-tag">Default Baseline</span>
                    )}
                  </div>
                </div>
                <div className="detail-actions">
                  <button
                    className="insights-action-btn"
                    onClick={() => handleExportSession(sessionDetail.session_id, 'csv')}
                  >
                    <Download size={14} /> CSV
                  </button>
                  <button
                    className="insights-action-btn"
                    onClick={() => handleExportSession(sessionDetail.session_id, 'json')}
                  >
                    <Download size={14} /> JSON
                  </button>
                  <button
                    className="insights-action-btn danger-btn"
                    onClick={() => handleDeleteSession(sessionDetail.session_id)}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>

              {/* Summary Cards */}
              <div className="session-metric-grid">
                <div className="metric-box">
                  <span className="label">Start Time</span>
                  <span className="value">{formatDateTime(sessionDetail.summary.start_time)}</span>
                </div>
                <div className="metric-box">
                  <span className="label">Duration</span>
                  <span className="value">{formatDuration(sessionDetail.summary.duration_sec)}</span>
                </div>
                <div className="metric-box">
                  <span className="label">Total Events</span>
                  <span className="value">{sessionDetail.summary.total_events}</span>
                </div>
                <div className="metric-box">
                  <span className="label">Drowsiness Alerts</span>
                  <span className={`value ${sessionDetail.summary.alert_count > 0 ? 'text-red' : 'text-green'}`}>
                    {sessionDetail.summary.alert_count}
                  </span>
                </div>
                <div className="metric-box">
                  <span className="label">Face Tracking Coverage</span>
                  <span className="value">{(sessionDetail.summary.tracking_coverage_ratio * 100).toFixed(1)}%</span>
                </div>
                <div className="metric-box">
                  <span className="label">Mean EAR / PERCLOS</span>
                  <span className="value">
                    {sessionDetail.summary.mean_ear.toFixed(3)} / {(sessionDetail.summary.mean_perclos * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Sparklines & Timeline Trends */}
              <div className="sparklines-container">
                <div className="sparkline-card">
                  <h4>Eye Aspect Ratio (EAR) Trajectory</h4>
                  <p className="sparkline-subtitle">
                    Dashed line marks the closure decision threshold (
                    {sessionDetail.summary.calibrated_threshold?.toFixed(3) || '0.250'}).
                  </p>
                  <EarSparkline
                    events={sessionDetail.events}
                    threshold={sessionDetail.summary.calibrated_threshold || 0.25}
                  />
                </div>

                <div className="sparkline-card">
                  <h4>PERCLOS Fatigue Index Trend</h4>
                  <p className="sparkline-subtitle">
                    Dashed line indicates 15.0% fatigue warning threshold over window.
                  </p>
                  <PerclosSparkline events={sessionDetail.events} />
                </div>
              </div>

              {/* Event Timeline Table */}
              <div className="timeline-table-wrapper">
                <h4>Event Timeline ({sessionDetail.events.length} records)</h4>
                <div className="table-responsive">
                  <table className="insights-table">
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Severity</th>
                        <th>Event Type</th>
                        <th>EAR</th>
                        <th>Closure</th>
                        <th>PERCLOS</th>
                        <th>Detection State</th>
                        <th>Notes / Context</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sessionDetail.events.map((ev, i) => (
                        <tr key={i} className={`row-severity-${ev.severity?.toLowerCase() || 'info'}`}>
                          <td className="monospace-cell">{formatTimestamp(ev.timestamp || ev.timestamp_iso)}</td>
                          <td>{renderSeverityBadge(ev.severity)}</td>
                          <td className="monospace-cell font-bold">{ev.event_type}</td>
                          <td className="monospace-cell">{ev.ear != null ? ev.ear.toFixed(3) : '—'}</td>
                          <td className="monospace-cell">{ev.closure_duration != null ? `${ev.closure_duration.toFixed(1)}s` : '—'}</td>
                          <td className="monospace-cell">{ev.perclos != null ? `${(ev.perclos * 100).toFixed(1)}%` : '—'}</td>
                          <td>{ev.detection_state || '—'}</td>
                          <td className="notes-cell">{ev.notes || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            /* Sessions List View */
            <div className="sessions-list-view">
              <div className="list-filter-bar">
                <div className="search-box">
                  <Search size={16} />
                  <input
                    type="text"
                    placeholder="Search session ID..."
                    value={sessionSearchText}
                    onChange={(e) => setSessionSearchText(e.target.value)}
                  />
                </div>
                <div className="list-stats">
                  Showing {filteredSessions.length} of {sessions.length} recorded sessions
                </div>
              </div>

              {filteredSessions.length === 0 ? (
                <div className="empty-state">
                  <History size={40} />
                  <h4>No sessions recorded yet</h4>
                  <p>Start driver monitoring on the Live Monitor tab to record events.</p>
                </div>
              ) : (
                <div className="sessions-grid">
                  {filteredSessions.map((s) => (
                    <div key={s.session_id} className="session-card">
                      <div className="card-top">
                        <div className="session-id-group">
                          <Clock size={16} />
                          <span className="session-id-text">{s.session_id}</span>
                        </div>
                        {renderQualityBadge(s.quality_status)}
                      </div>

                      <div className="card-metrics">
                        <div className="metric-row">
                          <span className="m-label">Recorded:</span>
                          <span className="m-val">{formatDateTime(s.start_time)}</span>
                        </div>
                        <div className="metric-row">
                          <span className="m-label">Duration:</span>
                          <span className="m-val">{formatDuration(s.duration_sec)}</span>
                        </div>
                        <div className="metric-row">
                          <span className="m-label">Alerts:</span>
                          <span className={`m-val ${s.alert_count > 0 ? 'alert-count-active' : ''}`}>
                            {s.alert_count} {s.alert_count === 1 ? 'alert' : 'alerts'}
                          </span>
                        </div>
                        <div className="metric-row">
                          <span className="m-label">Tracking:</span>
                          <span className="m-val">{(s.tracking_coverage_ratio * 100).toFixed(1)}%</span>
                        </div>
                        <div className="metric-row">
                          <span className="m-label">Baseline:</span>
                          <span className="m-val">
                            {s.calibrated
                              ? `Calibrated (${s.calibrated_threshold?.toFixed(3) || '0.250'})`
                              : 'Default (0.250)'}
                          </span>
                        </div>
                      </div>

                      <div className="card-footer">
                        <button
                          className="view-detail-btn"
                          onClick={() => loadSessionDetail(s.session_id)}
                        >
                          View Timeline <ChevronRight size={14} />
                        </button>
                        <div className="icon-actions">
                          <button
                            className="mini-icon-btn"
                            onClick={() => handleExportSession(s.session_id, 'csv')}
                            title="Export CSV"
                          >
                            <Download size={13} />
                          </button>
                          <button
                            className="mini-icon-btn danger"
                            onClick={() => handleDeleteSession(s.session_id)}
                            title="Delete Session"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* VIEW 2: LIVE LOGS */}
      {subView === 'live' && (
        <div className="insights-view-content">
          <div className="live-filter-bar">
            <div className="filter-item">
              <Filter size={16} />
              <label>Severity:</label>
              <select
                value={liveSeverityFilter}
                onChange={(e) => setLiveSeverityFilter(e.target.value)}
              >
                <option value="ALL">All Severities</option>
                <option value="CRITICAL">Critical Only</option>
                <option value="WARNING">Warning</option>
                <option value="INFO">Info</option>
                <option value="ERROR">Errors</option>
              </select>
            </div>

            <div className="search-box">
              <Search size={16} />
              <input
                type="text"
                placeholder="Filter by event type, notes, or session ID..."
                value={liveSearchText}
                onChange={(e) => setLiveSearchText(e.target.value)}
              />
            </div>

            <div className="live-status-indicator">
              <span className={`live-pulse ${isLivePaused ? 'paused' : 'running'}`} />
              <span>{isLivePaused ? 'Paused' : 'Streaming (2.5s poll)'}</span>
            </div>
          </div>

          <div className="table-responsive">
            <table className="insights-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Severity</th>
                  <th>Event Type</th>
                  <th>Session ID</th>
                  <th>EAR</th>
                  <th>PERCLOS</th>
                  <th>Tracking</th>
                  <th>State</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {filteredLiveEvents.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="empty-table-cell">
                      No live events match current filters.
                    </td>
                  </tr>
                ) : (
                  filteredLiveEvents.map((ev, i) => (
                    <tr key={i} className={`row-severity-${ev.severity?.toLowerCase() || 'info'}`}>
                      <td className="monospace-cell">{formatTimestamp(ev.timestamp || ev.timestamp_iso)}</td>
                      <td>{renderSeverityBadge(ev.severity)}</td>
                      <td className="monospace-cell font-bold">{ev.event_type}</td>
                      <td className="monospace-cell text-muted">{ev.session_id?.slice(-8) || '—'}</td>
                      <td className="monospace-cell">{ev.ear != null ? ev.ear.toFixed(3) : '—'}</td>
                      <td className="monospace-cell">{ev.perclos != null ? `${(ev.perclos * 100).toFixed(1)}%` : '—'}</td>
                      <td>{ev.tracking_state || '—'}</td>
                      <td>{ev.detection_state || '—'}</td>
                      <td className="notes-cell">{ev.notes || '—'}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* VIEW 3: PERSONALIZATION PROFILE */}
      {subView === 'profile' && (
        <div className="insights-view-content profile-view">
          <div className="profile-cards-grid">
            {/* Card 1: Baseline Parameters */}
            <div className="profile-card">
              <div className="card-header">
                <div className="title-group">
                  <Eye size={20} className="icon-cyan" />
                  <h4>Active Ocular Baseline Parameters</h4>
                </div>
                {profile?.is_active ? (
                  <span className="profile-status-badge badge-active">
                    <ShieldCheck size={14} /> Active Personalized Baseline
                  </span>
                ) : (
                  <span className="profile-status-badge badge-default">
                    <Sliders size={14} /> Factory Default (0.250)
                  </span>
                )}
              </div>

              <div className="profile-parameters">
                <div className="param-item">
                  <span className="param-label">Driver Identifier</span>
                  <span className="param-value font-mono">{profile?.driver_id || 'driver'}</span>
                </div>
                <div className="param-item">
                  <span className="param-label">Baseline Open-Eye EAR</span>
                  <span className="param-value font-mono">
                    {profile?.baseline_open_ear ? profile.baseline_open_ear.toFixed(3) : 'Not Calibrated (Default 0.312)'}
                  </span>
                </div>
                <div className="param-item highlight">
                  <span className="param-label">Personalized Eye-Closure Threshold</span>
                  <span className="param-value font-mono text-cyan">
                    {profile?.personalized_threshold ? profile.personalized_threshold.toFixed(3) : '0.250'}
                  </span>
                  <span className="param-subtext">
                    Calculated as 80% of open-eye EAR, clamped to safe boundaries.
                  </span>
                </div>
                <div className="param-item">
                  <span className="param-label">Safety Threshold Boundaries</span>
                  <span className="param-value font-mono">
                    [{profile?.min_threshold_bound?.toFixed(3) || '0.180'} – {profile?.max_threshold_bound?.toFixed(3) || '0.320'}]
                  </span>
                  <span className="param-subtext">
                    Prevents runaway hypersensitivity or unsafe missed closures.
                  </span>
                </div>
                <div className="param-item">
                  <span className="param-label">Profile Version & Detector</span>
                  <span className="param-value font-mono">
                    v{profile?.profile_version || 1} (Engine {profile?.detector_version || '2.1.0'})
                  </span>
                </div>
              </div>
            </div>

            {/* Card 2: Calibration Quality Metrics */}
            <div className="profile-card">
              <div className="card-header">
                <div className="title-group">
                  <ShieldCheck size={20} className="icon-green" />
                  <h4>Calibration Validation & Quality</h4>
                </div>
                {renderQualityBadge(profile?.quality_status || 'UNAVAILABLE')}
              </div>

              <div className="profile-parameters">
                <div className="param-item">
                  <span className="param-label">Last Calibration Time</span>
                  <span className="param-value">
                    {profile?.calibration_timestamp
                      ? new Date(profile.calibration_timestamp * 1000).toLocaleString()
                      : 'Never (Using Default Factory Baseline)'}
                  </span>
                </div>
                <div className="param-item">
                  <span className="param-label">Validated Samples Collected</span>
                  <span className="param-value font-mono">
                    {profile?.sample_count || 0} samples (Minimum required: 50)
                  </span>
                </div>
                <div className="param-item">
                  <span className="param-label">Observation Duration</span>
                  <span className="param-value font-mono">
                    {profile?.duration_sec ? `${profile.duration_sec.toFixed(1)}s` : '0.0s'} (Target: 10.0s)
                  </span>
                </div>
                <div className="param-item">
                  <span className="param-label">Facial Landmark Tracking Ratio</span>
                  <span className="param-value font-mono">
                    {profile?.tracking_ratio ? `${(profile.tracking_ratio * 100).toFixed(1)}%` : '0.0%'} (Min: 80%)
                  </span>
                </div>
                <div className="param-item">
                  <span className="param-label">Update Reason / Audit Trail</span>
                  <span className="param-value text-muted">
                    {profile?.update_reason || 'Default uncalibrated profile'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Personalization Actions */}
          <div className="profile-actions-panel">
            <div className="action-info">
              <h4>Driver Baseline Management</h4>
              <p>
                Personalization adapts detection sensitivity to individual ocular geometry, reducing false alarms
                from naturally narrower eyes without compromising safety. You can recalibrate anytime or reset
                to factory defaults.
              </p>
            </div>
            <div className="action-buttons">
              {onSwitchToMonitor && (
                <button className="primary-action-btn" onClick={onSwitchToMonitor}>
                  <ExternalLink size={16} /> Go to Live Monitor to Calibrate
                </button>
              )}
              <button
                className="danger-action-btn"
                onClick={handleResetProfile}
                disabled={!profile?.is_active}
              >
                <Trash2 size={16} /> Reset to Factory Default
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SVG Sparkline Visualizations
// ---------------------------------------------------------------------------

function EarSparkline({ events, threshold }: { events: LogEvent[]; threshold: number }) {
  const validData = useMemo(() => {
    return events
      .filter((e) => e.ear != null && !isNaN(e.ear) && e.ear > 0)
      .map((e, index) => ({
        index,
        ear: e.ear as number,
        severity: e.severity,
        timestamp: e.timestamp || e.timestamp_iso || ''
      }));
  }, [events]);

  if (validData.length < 2) {
    return <div className="sparkline-empty">Insufficient continuous EAR data points in this session.</div>;
  }

  const width = 600;
  const height = 120;
  const padX = 40;
  const padY = 15;

  const minEar = 0.05;
  const maxEar = 0.45;

  const scaleX = (idx: number) => padX + (idx / (validData.length - 1)) * (width - 2 * padX);
  const scaleY = (val: number) => {
    const clamped = Math.max(minEar, Math.min(maxEar, val));
    return height - padY - ((clamped - minEar) / (maxEar - minEar)) * (height - 2 * padY);
  };

  const polylinePoints = validData.map((d) => `${scaleX(d.index)},${scaleY(d.ear)}`).join(' ');
  const thresholdY = scaleY(threshold);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="sparkline-svg" preserveAspectRatio="none">
      {/* Grid lines */}
      <line x1={padX} y1={scaleY(0.1)} x2={width - padX} y2={scaleY(0.1)} stroke="rgba(255,255,255,0.08)" strokeDasharray="2,2" />
      <line x1={padX} y1={scaleY(0.2)} x2={width - padX} y2={scaleY(0.2)} stroke="rgba(255,255,255,0.08)" strokeDasharray="2,2" />
      <line x1={padX} y1={scaleY(0.3)} x2={width - padX} y2={scaleY(0.3)} stroke="rgba(255,255,255,0.08)" strokeDasharray="2,2" />
      <line x1={padX} y1={scaleY(0.4)} x2={width - padX} y2={scaleY(0.4)} stroke="rgba(255,255,255,0.08)" strokeDasharray="2,2" />

      {/* Axis text */}
      <text x={padX - 8} y={scaleY(0.3) + 3} textAnchor="end" className="svg-label">0.30</text>
      <text x={padX - 8} y={scaleY(0.2) + 3} textAnchor="end" className="svg-label">0.20</text>
      <text x={padX - 8} y={scaleY(0.1) + 3} textAnchor="end" className="svg-label">0.10</text>

      {/* Threshold reference line */}
      <line
        x1={padX}
        y1={thresholdY}
        x2={width - padX}
        y2={thresholdY}
        stroke="#f59e0b"
        strokeWidth="1.5"
        strokeDasharray="4,3"
      />
      <text x={width - padX + 5} y={thresholdY + 3} className="svg-threshold-label">
        Thresh ({threshold.toFixed(2)})
      </text>

      {/* EAR trajectory polyline */}
      <polyline fill="none" stroke="#38bdf8" strokeWidth="2" points={polylinePoints} />

      {/* Critical / Warning Event markers */}
      {validData
        .filter((d) => d.severity === 'CRITICAL' || d.severity === 'WARNING')
        .map((d, i) => (
          <circle
            key={i}
            cx={scaleX(d.index)}
            cy={scaleY(d.ear)}
            r={d.severity === 'CRITICAL' ? 4.5 : 3.5}
            fill={d.severity === 'CRITICAL' ? '#ef4444' : '#f59e0b'}
            stroke="#0f172a"
            strokeWidth="1.5"
          />
        ))}
    </svg>
  );
}

function PerclosSparkline({ events }: { events: LogEvent[] }) {
  const validData = useMemo(() => {
    return events
      .filter((e) => e.perclos != null && !isNaN(e.perclos))
      .map((e, index) => ({
        index,
        perclos: e.perclos as number,
        timestamp: e.timestamp || e.timestamp_iso || ''
      }));
  }, [events]);

  if (validData.length < 2) {
    return <div className="sparkline-empty">Insufficient continuous PERCLOS data points in this session.</div>;
  }

  const width = 600;
  const height = 120;
  const padX = 40;
  const padY = 15;

  const minPerclos = 0.0;
  const maxPerclos = 0.4; // 40% top range

  const scaleX = (idx: number) => padX + (idx / (validData.length - 1)) * (width - 2 * padX);
  const scaleY = (val: number) => {
    const clamped = Math.max(minPerclos, Math.min(maxPerclos, val));
    return height - padY - ((clamped - minPerclos) / (maxPerclos - minPerclos)) * (height - 2 * padY);
  };

  const polylinePoints = validData.map((d) => `${scaleX(d.index)},${scaleY(d.perclos)}`).join(' ');
  const thresholdY = scaleY(0.15); // 15% threshold

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="sparkline-svg" preserveAspectRatio="none">
      {/* Grid lines */}
      <line x1={padX} y1={scaleY(0.1)} x2={width - padX} y2={scaleY(0.1)} stroke="rgba(255,255,255,0.08)" strokeDasharray="2,2" />
      <line x1={padX} y1={scaleY(0.2)} x2={width - padX} y2={scaleY(0.2)} stroke="rgba(255,255,255,0.08)" strokeDasharray="2,2" />
      <line x1={padX} y1={scaleY(0.3)} x2={width - padX} y2={scaleY(0.3)} stroke="rgba(255,255,255,0.08)" strokeDasharray="2,2" />

      {/* Axis text */}
      <text x={padX - 8} y={scaleY(0.3) + 3} textAnchor="end" className="svg-label">30%</text>
      <text x={padX - 8} y={scaleY(0.2) + 3} textAnchor="end" className="svg-label">20%</text>
      <text x={padX - 8} y={scaleY(0.1) + 3} textAnchor="end" className="svg-label">10%</text>

      {/* 15% Warning threshold line */}
      <line
        x1={padX}
        y1={thresholdY}
        x2={width - padX}
        y2={thresholdY}
        stroke="#f43f5e"
        strokeWidth="1.5"
        strokeDasharray="4,3"
      />
      <text x={width - padX + 5} y={thresholdY + 3} className="svg-threshold-label danger">
        15% Limit
      </text>

      {/* PERCLOS trajectory polyline */}
      <polyline fill="none" stroke="#c084fc" strokeWidth="2" points={polylinePoints} />
    </svg>
  );
}
