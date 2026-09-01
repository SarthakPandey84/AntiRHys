# 🚗 AntiRHys: Driver Fatigue & Road Hypnosis Detection

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

**AntiRHys** is a real-time, full-stack web application meticulously designed to detect driver fatigue and prevent road hypnosis. 

By capturing a live webcam feed and securely streaming it via WebSockets to a high-performance backend, the system continuously analyzes the driver's facial landmarks. Using the **Eye Aspect Ratio (EAR)**, it triggers robust visual and audible alerts when prolonged eye closure is detected, ensuring the driver stays awake and safe.

---

## ✨ Key Features

- **👀 Real-Time Monitoring:** Low-latency WebSocket streaming for instant fatigue detection.
- **🚨 Intelligent Alerts:** Visual dashboard warnings combined with an unmissable **Audible Alarm** when drowsiness is detected for an extended period.
- **📊 Session Data Logging:** Automatically records blink counts, alert events, and drowsiness timestamps into a CSV file for post-session analysis.
- **🛡️ Robust State Tracking:** Ensures false positives are minimized and sustained drowsiness triggers a persistent alert mechanism.
- **🎨 Premium UI/UX:** A stunning glassmorphism, dark-mode dashboard tailored for seamless interaction.

---

## 🛠️ Tech Stack

### Backend
- **Python 3** & **FastAPI**
- **WebSockets** for real-time video frame transmission
- **OpenCV** & **MediaPipe Face Mesh** for precise facial landmark detection
- **SoundDevice** & **NumPy** for audible alerts

### Frontend
- **React (TypeScript)** built with **Vite**
- **HTML Canvas** for frame rendering
- **Lucide React** for crisp, modern iconography
- Vanilla CSS with a polished **Glassmorphism Aesthetic**

---

## 🚀 Getting Started

### Prerequisites
- [Node.js](https://nodejs.org/) (v18+ recommended)
- [Python](https://www.python.org/) (3.9+)

### 1. Backend Setup

1. Open a terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   # source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   *The WebSocket will now listen on `ws://localhost:8000/ws/detect`.*

### 2. Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the Node modules:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```

---

## 🚦 Usage Guide

1. Navigate to the Vite URL (typically `http://localhost:5173`) in your browser.
2. Grant the application permission to access your webcam.
3. Click the **"Start Camera"** button to initiate tracking.
4. The system will start calculating your EAR in real-time. If you close your eyes for an extended period, the dashboard will flash red, sound an alarm, and log the event.
5. Click **"Stop Camera"** to end the session. The backend will automatically finalize and save the session log as a `.csv` file in the `backend` directory.

---

*Stay alert, stay safe.* 🛣️
