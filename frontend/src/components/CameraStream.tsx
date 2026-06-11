import React, { useRef, useEffect, useState } from 'react';
import { Video, VideoOff } from 'lucide-react';
import type { DetectionState } from '../App';

interface CameraStreamProps {
  onDetectionUpdate: (state: DetectionState) => void;
  landmarks: { x: number; y: number }[];
}

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/detect';

const CameraStream: React.FC<CameraStreamProps> = ({ onDetectionUpdate, landmarks }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const extractCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  
  const [isStreaming, setIsStreaming] = useState(false);

  // Initialize WebSocket
  useEffect(() => {
    wsRef.current = new WebSocket(WS_URL);
    
    wsRef.current.onopen = () => {
      console.log('WebSocket Connected');
    };
    
    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onDetectionUpdate(data);
      } catch (err) {
        console.error('Error parsing WS message', err);
      }
    };
    
    wsRef.current.onclose = () => {
      console.log('WebSocket Disconnected');
    };

    return () => {
      wsRef.current?.close();
    };
  }, [onDetectionUpdate]);

  // Start Camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Could not access camera. Please allow permissions.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  };

  // Capture Frame Loop
  useEffect(() => {
    let intervalId: number;
    
    if (isStreaming) {
      intervalId = window.setInterval(() => {
        if (videoRef.current && extractCanvasRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
          const video = videoRef.current;
          const canvas = extractCanvasRef.current;
          const ctx = canvas.getContext('2d');
          
          if (ctx && video.videoWidth > 0) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Get base64 jpeg
            const base64Data = canvas.toDataURL('image/jpeg', 0.7);
            wsRef.current.send(base64Data);
          }
        }
      }, 100); // ~10 FPS
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isStreaming]);

  // Draw Landmarks
  useEffect(() => {
    if (!overlayCanvasRef.current || !videoRef.current) return;
    const canvas = overlayCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas size to video size
    canvas.width = videoRef.current.clientWidth;
    canvas.height = videoRef.current.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (landmarks && landmarks.length > 0) {
      ctx.fillStyle = '#00ff00';
      landmarks.forEach((point) => {
        // Point is normalized [0, 1]
        const x = point.x * canvas.width;
        const y = point.y * canvas.height;
        
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
  }, [landmarks, isStreaming]);

  return (
    <div className="stream-container glass-panel">
      <div className="video-wrapper">
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          className="video-feed"
        />
        <canvas 
          ref={overlayCanvasRef} 
          className="overlay-canvas"
        />
      </div>
      
      {/* Hidden canvas for extraction */}
      <canvas ref={extractCanvasRef} style={{ display: 'none' }} />
      
      <div className="controls">
        {!isStreaming ? (
          <button className="btn btn-primary" onClick={startCamera}>
            <Video size={20} /> Start Camera
          </button>
        ) : (
          <button className="btn btn-danger" onClick={stopCamera}>
            <VideoOff size={20} /> Stop Camera
          </button>
        )}
      </div>
    </div>
  );
};

export default CameraStream;
