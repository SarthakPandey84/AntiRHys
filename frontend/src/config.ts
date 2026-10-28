/**
 * AntiRHys Configuration
 *
 * Safely resolves the WebSocket backend URL with protocol validation:
 * - Falls back to ws://localhost:8000/ws/detect for local development
 * - Automatically requires wss:// if frontend is hosted over https:// (e.g. Vercel)
 * - Attaches authentication token (?token=...) when VITE_AUTH_TOKEN is configured
 */

export function getWebSocketUrl(): string {
  const envUrl = import.meta.env.VITE_WS_URL;
  const authToken = import.meta.env.VITE_AUTH_TOKEN;
  const isHttps = typeof window !== 'undefined' && window.location.protocol === 'https:';

  let resolvedUrl: string;
  if (envUrl && typeof envUrl === 'string' && envUrl.trim().length > 0) {
    let cleanUrl = envUrl.trim();
    if (isHttps && cleanUrl.startsWith('ws://')) {
      console.warn('[AntiRHys Config] Upgrading ws:// to wss:// to avoid browser mixed-content blockage on HTTPS.');
      cleanUrl = cleanUrl.replace(/^ws:\/\//, 'wss://');
    }
    resolvedUrl = cleanUrl;
  } else {
    // Fallback: local development
    resolvedUrl = isHttps ? 'wss://localhost:8000/ws/detect' : 'ws://localhost:8000/ws/detect';
  }

  if (authToken && typeof authToken === 'string' && authToken.trim().length > 0) {
    const separator = resolvedUrl.includes('?') ? '&' : '?';
    resolvedUrl = `${resolvedUrl}${separator}token=${encodeURIComponent(authToken.trim())}`;
  }

  return resolvedUrl;
}
