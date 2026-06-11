import cv2
import mediapipe as mp
import numpy as np
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import math
import time
import csv
from datetime import datetime
import threading
try:
    import sounddevice as sd
    has_audio_device = True
except Exception as e:
    print(f"Warning: sounddevice could not be imported (expected on headless servers): {e}")
    has_audio_device = False

app = FastAPI(title="Fatigue Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe landmark indices for eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES_THRESHOLD = 15

# Backend Alarm Logic
alarm_thread = None

def play_alarm_sound():
    if not has_audio_device:
        return
    duration = 10.0  # seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Alternating 800 Hz and 1000 Hz every 0.5s for a siren effect
    freq = np.where((t % 1.0) < 0.5, 800, 1000)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    # Loud sine wave
    wave = 0.8 * np.sin(phase)
    try:
        sd.play(wave, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Audio playback error: {e}")

def trigger_alarm():
    global alarm_thread
    if alarm_thread is None or not alarm_thread.is_alive():
        alarm_thread = threading.Thread(target=play_alarm_sound)
        alarm_thread.start()

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(eye_landmarks):
    # p1, p2, p3, p4, p5, p6
    # eye_landmarks is a list of 6 (x,y) tuples
    # EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    
    dist_p2_p6 = euclidean_distance(p2, p6)
    dist_p3_p5 = euclidean_distance(p3, p5)
    dist_p1_p4 = euclidean_distance(p1, p4)
    
    if dist_p1_p4 == 0:
        return 0.0
    
    ear = (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)
    return ear

@app.websocket("/ws/detect")
async def detect_fatigue(websocket: WebSocket):
    await websocket.accept()
    
    consecutive_closed_frames = 0
    blink_count = 0
    is_drowsy = False
    was_closed = False
    drowsy_trigger_time = 0
    last_logged_state = "Alert"
    
    try:
        while True:
            data = await websocket.receive_text()
            # data is base64 string
            if "," in data:
                base64_str = data.split(",")[1]
            else:
                base64_str = data
                
            img_data = base64.b64decode(base64_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            # Process with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            response_payload = {
                "ear": 0.0,
                "state": "Alert",
                "blink_count": blink_count,
                "landmarks": []
            }
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                
                # Extract coordinates
                left_eye_coords = []
                for idx in LEFT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    left_eye_coords.append((landmark.x * w, landmark.y * h))
                    
                right_eye_coords = []
                for idx in RIGHT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    right_eye_coords.append((landmark.x * w, landmark.y * h))
                    
                left_ear = calculate_ear(left_eye_coords)
                right_ear = calculate_ear(right_eye_coords)
                
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Prepare landmarks to send back for visualization (normalized coordinates)
                all_eye_indices = LEFT_EYE + RIGHT_EYE
                landmarks_list = [{"x": face_landmarks.landmark[i].x, "y": face_landmarks.landmark[i].y} for i in all_eye_indices]
                
                # Logic for drowsiness and blinking
                if avg_ear < EAR_THRESHOLD:
                    consecutive_closed_frames += 1
                    if not was_closed:
                        was_closed = True
                else:
                    if was_closed:
                        # Eye just opened, count as a blink if it wasn't a long closure
                        if consecutive_closed_frames < CONSECUTIVE_FRAMES_THRESHOLD:
                            blink_count += 1
                        was_closed = False
                    consecutive_closed_frames = 0
                
                # Check Drowsy hold time (10 seconds)
                if consecutive_closed_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                    is_drowsy = True
                    drowsy_trigger_time = time.time()
                else:
                    if is_drowsy and (time.time() - drowsy_trigger_time < 10.0):
                        is_drowsy = True
                    else:
                        is_drowsy = False
                current_state = "Drowsy" if is_drowsy else "Alert"
                if current_state == "Drowsy":
                    trigger_alarm()
                
                # Logging state changes
                if current_state != last_logged_state:
                    with open("session_log.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now().isoformat(), current_state, avg_ear, blink_count])
                    last_logged_state = current_state
                
                response_payload["ear"] = round(avg_ear, 3)
                response_payload["state"] = current_state
                response_payload["blink_count"] = blink_count
                response_payload["landmarks"] = landmarks_list
            else:
                # If face is lost, still enforce the 10 second hold for drowsiness
                if is_drowsy and (time.time() - drowsy_trigger_time < 10.0):
                    is_drowsy = True
                else:
                    is_drowsy = False
                    
                current_state = "Drowsy" if is_drowsy else "Alert"
                if current_state == "Drowsy":
                    trigger_alarm()
                    
                response_payload["state"] = current_state
                
            await websocket.send_json(response_payload)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
