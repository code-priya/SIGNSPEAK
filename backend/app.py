import asyncio
import json
import base64
import cv2
import numpy as np
import websockets
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time
import signal
import sys

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load model and metadata
try:
    model = tf.keras.models.load_model('models/cnn_lstm_model.h5', compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
    ACTIONS = np.array(metadata['actions'])
    SEQUENCE_LENGTH = metadata['sequence_length']
    print(f"✅ Model loaded with {len(ACTIONS)} gestures")
    print(f"   Gestures: {', '.join(ACTIONS)}")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("Using fallback mode with basic gestures")
    ACTIONS = np.array(['hello', 'thanks', 'iloveyou', 'yes', 'no'])
    SEQUENCE_LENGTH = 30

# Detection parameters
sequence = deque(maxlen=SEQUENCE_LENGTH)
sentence = []
predictions_history = []
current_mode = "standard"
confidence_threshold = 0.7
frame_count = 0
performance_stats = {"fps": 0, "avg_inference_time": 0}

class SignLanguageDetector:
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.predictions_history = []
        self.last_prediction = None
        self.last_prediction_time = 0
        self.cooldown_ms = 500  # Prevent rapid-fire predictions
        
    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    
    def process_frame(self, frame, mode="standard", threshold=0.7):
        global frame_count, performance_stats
        
        start_time = time.time()
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = self.mediapipe_detection(frame, holistic)
            keypoints = self.extract_keypoints(results)
            
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-SEQUENCE_LENGTH:]
            
            result = {
                "prediction": None,
                "confidence": 0,
                "sentence": self.sentence.copy(),
                "confidence_scores": [0] * len(ACTIONS),
                "mode": mode,
                "fps": performance_stats["fps"]
            }
            
            if len(self.sequence) == SEQUENCE_LENGTH:
                try:
                    # Predict using model
                    res = model.predict(np.expand_dims(np.array(self.sequence), axis=0), verbose=0)[0]
                    
                    # Apply mode-specific smoothing
                    if mode == "advanced":
                        self.predictions_history.append(np.argmax(res))
                        if len(self.predictions_history) > 10:
                            self.predictions_history.pop(0)
                        gesture_idx = max(set(self.predictions_history), key=self.predictions_history.count) if self.predictions_history else np.argmax(res)
                    elif mode == "batch":
                        if len(self.predictions_history) < 15:
                            self.predictions_history.append(np.argmax(res))
                            gesture_idx = np.argmax(res)
                        else:
                            gesture_idx = max(set(self.predictions_history), key=self.predictions_history.count)
                            self.predictions_history = []
                    else:  # standard
                        gesture_idx = np.argmax(res)
                    
                    confidence = res[gesture_idx]
                    gesture = ACTIONS[gesture_idx]
                    
                    result["confidence_scores"] = res.tolist()
                    
                    # Check cooldown and threshold
                    current_time = time.time() * 1000
                    if (confidence > threshold and 
                        (current_time - self.last_prediction_time) > self.cooldown_ms):
                        
                        result["prediction"] = gesture
                        result["confidence"] = float(confidence)
                        
                        if len(self.sentence) == 0 or self.sentence[-1] != gesture:
                            self.sentence.append(gesture)
                            if len(self.sentence) > 5:
                                self.sentence.pop(0)
                        
                        self.last_prediction = gesture
                        self.last_prediction_time = current_time
                    
                    result["sentence"] = self.sentence.copy()
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            # Update FPS stats
            inference_time = (time.time() - start_time) * 1000
            performance_stats["avg_inference_time"] = performance_stats["avg_inference_time"] * 0.95 + inference_time * 0.05
            frame_count += 1
            if frame_count % 30 == 0:
                performance_stats["fps"] = int(1000 / performance_stats["avg_inference_time"]) if performance_stats["avg_inference_time"] > 0 else 0
            
            return result

detector = SignLanguageDetector()

async def handle_connection(websocket, path):
    global current_mode, confidence_threshold
    
    print(f"✅ Client connected from {websocket.remote_address}")
    
    try:
        async for message in websocket:
            try:
                # Handle JSON config
                if isinstance(message, str) and message.startswith('{'):
                    config = json.loads(message)
                    if "mode" in config:
                        current_mode = config["mode"]
                        print(f"🔄 Mode changed to: {current_mode}")
                    if "threshold" in config:
                        confidence_threshold = config["threshold"]
                        print(f"🎯 Threshold changed to: {confidence_threshold}")
                    continue
                
                # Handle image data
                if isinstance(message, str) and message.startswith('data:image'):
                    image_data = message.split(',')[1]
                    img_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    nparr = np.frombuffer(message, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None and frame.size > 0:
                    result = detector.process_frame(frame, current_mode, confidence_threshold)
                    await websocket.send(json.dumps(result))
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    # Start server
    async with websockets.serve(handle_connection, "0.0.0.0", 8765, ping_interval=20, ping_timeout=60):
        print("=" * 60)
        print("🤟 SignSpeak Backend Server (CNN+LSTM Hybrid)")
        print("=" * 60)
        print(f"✅ WebSocket server: ws://localhost:8765")
        print(f"📊 Gestures loaded: {len(ACTIONS)}")
        print(f"🎯 Detection modes: standard, advanced, batch")
        print(f"🔧 Confidence threshold: {confidence_threshold}")
        print("=" * 60)
        print("\n📡 To expose to internet:")
        print("   ngrok http 8765")
        print("\n🎮 Ready for connections...")
        print("=" * 60)
        
        await asyncio.Future()

def signal_handler(sig, frame):
    print("\n\n👋 Shutting down server...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    asyncio.run(main())
