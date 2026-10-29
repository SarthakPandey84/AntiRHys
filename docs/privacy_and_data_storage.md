# AntiRHys Privacy, Data Sovereignty & Storage Architecture

## 1. Overview & Privacy Principles
AntiRHys is designed with **Privacy by Design** and strict **Data Minimization** principles. Driver monitoring must protect the operator's privacy while delivering dependable vigilance assistance.

### Key Tenets:
1. **No Raw Video or Imagery Storage:** AntiRHys processes camera frames strictly in transient memory. Raw video streams, decoded frames, and high-dimensional biometric facial meshes are immediately discarded after landmark extraction and are **never** written to persistent disk storage or uploaded to external clouds.
2. **Strictly Scalar Telemetry:** Logged events contain only aggregated scalar metrics (Eye Aspect Ratio, PERCLOS, head pose Euler angles, blink counts, timestamps, and detection state enums).
3. **No Personal Identifiers:** Sessions and personalization profiles use anonymous pseudonymous handles (`session_YYYYMMDD_HHMMSS_<hex>`, `driver`). No names, emails, biometrics, or vehicle identification numbers are collected or stored.
4. **Complete Data Sovereignty:** Users retain full control over their telemetry: they can inspect every event, export data in open formats (CSV/JSON), and permanently delete individual sessions or entire historical records at any time.

---

## 2. Storage Architecture by Deployment Environment

| Storage Aspect | Local Native Deployment | Ephemeral Cloud Hosting (e.g. Render / Vercel) |
| :--- | :--- | :--- |
| **Session Log File** | `session_log.csv` (persistent local disk) | Ephemeral disk container; automatically cleared when container restarts or redeploys |
| **Driver Profile** | `driver_profile.json` (persistent local disk) | Ephemeral disk container; reset to factory default upon container recycling unless external storage is mounted |
| **Live Ring Buffer** | In-memory `collections.deque(maxlen=250)` | In-memory `collections.deque(maxlen=250)` |
| **Disk Bound** | Strictly capped at 10 MB (`MAX_LOG_FILE_BYTES`) with in-place FIFO record pruning | Strictly capped at 10 MB with in-place FIFO record pruning |

> [!NOTE]
> On platforms like Render and Vercel free tiers, the file system is ephemeral. Any session logs or calibration profiles stored on disk will be automatically cleared whenever the service spins down or re-deploys. For persistent local deployments, data remains under the local operator's control on their personal machine.

---

## 3. Data Schema & Retention Bounds

### 3.1 Session Log Retention & Pruning Policy
To prevent uncontrolled disk consumption:
- The session log file size is checked prior to writing each event.
- When `session_log.csv` exceeds **10 MB** (`MAX_LOG_FILE_BYTES = 10 * 1024 * 1024`), an in-place FIFO pruning operation is executed: the CSV header and the newest 60% of entries are preserved, while the oldest 40% of records are pruned.
- This ensures the log stays strictly bounded under 10 MB without disk exhaustion or unbounded growth, operating entirely in-place without generating auxiliary backup files.

### 3.2 In-Memory Live Ring Buffer
- A bounded FIFO ring buffer stores the most recent **250 events** in memory (`collections.deque(maxlen=250)`).
- This enables responsive, near-real-time streaming to the "Live Logs" tab without repeated disk I/O.
- The ring buffer is protected by a threading lock (`threading.Lock`) for thread-safe concurrent access.

---

## 4. User Control & Data Deletion Capabilities

AntiRHys provides explicit REST endpoints and UI controls for data management:

1. **Inspect Sessions:** `GET /api/sessions` lists all recorded driving sessions with duration, alert count, and quality status.
2. **View Timeline:** `GET /api/sessions/{session_id}` provides the full chronological event log and scalar trajectory for that drive.
3. **Export Single Session:** `GET /api/sessions/{session_id}/export?format=csv` or `?format=json` downloads a standardized export file.
4. **Export All Sessions:** `GET /api/export/all?format=csv` or `?format=json` exports the complete log history.
5. **Delete Specific Session:** `DELETE /api/sessions/{session_id}` removes all records associated with that session and cleans disk storage.
6. **Clear All History:** `DELETE /api/sessions` wipes all session records and log files immediately.
7. **Reset Personalization:** `POST /api/profile/reset` deletes custom ocular baselines and restores the safe factory threshold (`0.250`).

---

## 5. Security & Access Controls

1. **Origin Verification (CORS & CSWSH Defense):**
   - Both WebSocket connections (`/ws/detect`) and HTTP endpoints (`/api/*`) validate the incoming `Origin` header against the configured allowlist (`ALLOWED_ORIGINS`).
   - Prevents Cross-Site WebSocket Hijacking (CSWSH) and cross-origin telemetry exfiltration from untrusted browser origins.
2. **Constant-Time Token Authentication:**
   - When `AUTH_TOKEN` is configured, all WebSocket connections and HTTP endpoints require a valid token via query parameter (`?token=...`) or header (`Authorization: Bearer <token>`).
   - Verification uses `hmac.compare_digest` to eliminate timing-attack vulnerabilities.
3. **Rate Limiting & Payload Bounds:**
   - Raw WebSocket message payloads are capped at 5 MB (`MAX_MESSAGE_BYTES`).
   - Frame processing is throttled to maximum ~30 FPS (`MIN_FRAME_INTERVAL_SEC = 0.03`) to eliminate denial-of-service queue build-up.

