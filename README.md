# Driver Fatigue and Road Hypnosis Detection Prototype

A real-time full-stack web application designed to detect driver fatigue and road hypnosis. It uses a React frontend to capture a live webcam feed and stream it via WebSockets to a FastAPI backend. The backend processes the frames using OpenCV and MediaPipe Face Mesh to calculate the Eye Aspect Ratio (EAR), triggering alerts when prolonged eye closure is detected.

## Tech Stack
- **Backend**: Python, FastAPI, WebSockets, OpenCV, MediaPipe
- **Frontend**: React (TypeScript), Vite, HTML Canvas, Lucide React
- **Styling**: Vanilla CSS with a modern dark-mode, glassmorphism aesthetic

## Prerequisites
- Node.js (v18+ recommended)
- Python 3.9+

## Installation & Setup

### 1. Backend Setup
1. Open a terminal and navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   # source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   The backend WebSocket will now be listening on `ws://localhost:8000/ws/detect`.

### 2. Frontend Setup
1. Open a new terminal and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install the dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```

## Usage
1. Open the URL provided by Vite (typically `http://localhost:5173`) in your browser.
2. Allow the browser to access your camera.
3. Click "Start Camera".
4. The backend will begin returning real-time EAR values. If you close your eyes for an extended period, the dashboard will flash red and alert you that the driver is sleeping.
