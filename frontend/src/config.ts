/**
 * AntiRHys Configuration
 *
 * Safely resolves the WebSocket backend URL with protocol validation:
 * - Falls back to ws://localhost:8000/ws/detect for local development
 * - Automatically defaults to the live Render backend (wss://antirhys-backend.onrender.com/ws/detect)
 *   when deployed on Vercel or any remote HTTPS host without custom environment variables
 * - Automatically requires wss:// if frontend is hosted over https:// (e.g. Vercel)
 * - Attaches authentication token (?token=...) when VITE_AUTH_TOKEN is configured
 */

export function getWebSocketUrl(): string {
  const envUrl = import.meta.env.VITE_WS_URL;
  const authToken = import.meta.env.VITE_AUTH_TOKEN;
  const isHttps = typeof window !== 'undefined' && window.location.protocol === 'https:';
  const isRemote = typeof window !== 'undefined' &&
    window.location.hostname !== 'localhost' &&
    window.location.hostname !== '127.0.0.1';

  let resolvedUrl: string;
  if (envUrl && typeof envUrl === 'string' && envUrl.trim().length > 0) {
    let cleanUrl = envUrl.trim();
    if (isHttps && cleanUrl.startsWith('ws://')) {
      console.warn('[AntiRHys Config] Upgrading ws:// to wss:// to avoid browser mixed-content blockage on HTTPS.');
      cleanUrl = cleanUrl.replace(/^ws:\/\//, 'wss://');
    }
    resolvedUrl = cleanUrl;
  } else if (isRemote) {
    // When hosted remotely (e.g. on Vercel), default to the public Render backend service
    resolvedUrl = 'wss://antirhys-backend.onrender.com/ws/detect';
  } else {
    // Local development fallback
    resolvedUrl = isHttps ? 'wss://localhost:8000/ws/detect' : 'ws://localhost:8000/ws/detect';
  }

  if (authToken && typeof authToken === 'string' && authToken.trim().length > 0) {
    const separator = resolvedUrl.includes('?') ? '&' : '?';
    resolvedUrl = `${resolvedUrl}${separator}token=${encodeURIComponent(authToken.trim())}`;
  }

  return resolvedUrl;
}

export function getAuthToken(): string | null {
  const token = import.meta.env.VITE_AUTH_TOKEN;
  return (token && typeof token === 'string' && token.trim().length > 0) ? token.trim() : null;
}

export function getApiBaseUrl(): string {
  const envApiUrl = import.meta.env.VITE_API_URL;
  const isHttps = typeof window !== 'undefined' && window.location.protocol === 'https:';
  const isRemote = typeof window !== 'undefined' &&
    window.location.hostname !== 'localhost' &&
    window.location.hostname !== '127.0.0.1';

  if (envApiUrl && typeof envApiUrl === 'string' && envApiUrl.trim().length > 0) {
    let cleanUrl = envApiUrl.trim().replace(/\/+$/, '');
    if (isHttps && cleanUrl.startsWith('http://')) {
      cleanUrl = cleanUrl.replace(/^http:\/\//, 'https://');
    }
    return cleanUrl;
  }

  // Derive from WebSocket URL base
  const wsUrl = getWebSocketUrl();
  try {
    const parsed = new URL(wsUrl);
    const protocol = parsed.protocol === 'wss:' ? 'https:' : 'http:';
    return `${protocol}//${parsed.host}`;
  } catch {
    if (isRemote) {
      return 'https://antirhys-backend.onrender.com';
    }
    return isHttps ? 'https://localhost:8000' : 'http://localhost:8000';
  }
}
