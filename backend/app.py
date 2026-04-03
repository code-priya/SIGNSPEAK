import asyncio
import json
import base64
import cv2
import numpy as np
import websockets
import mediapipe as mp
import tensorflow as tf
from collections import deque
import threading
import time
import ssl

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load trained model
try:
    model = tf.keras.models.load_model('models/model_weights.h5', compile=False)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Model not found: {e}")
    print("Creating a dummy model for testing...")
    # Create a simple model for testing if no trained model exists
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, 1662)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Gesture mapping
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Detection parameters
sequence_length = 30
sequence = deque(maxlen=sequence_length)
sentence = []
predictions = []
threshold = 0.7

# Mode settings
current_mode = "standard"
confidence_threshold = 0.7

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def process_frame(frame, mode="standard"):
    global sequence, sentence, predictions, confidence_threshold
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        
        if len(sequence) == sequence_length:
            try:
                res = model.predict(np.expand_dims(np.array(sequence), axis=0), verbose=0)[0]
                
                if mode == "advanced":
                    predictions.append(np.argmax(res))
                    if len(predictions) > 10:
                        predictions.pop(0)
                    gesture_idx = max(set(predictions), key=predictions.count) if predictions else np.argmax(res)
                elif mode == "batch":
                    if len(predictions) < 10:
                        predictions.append(np.argmax(res))
                        gesture_idx = np.argmax(res)
                    else:
                        gesture_idx = max(set(predictions), key=predictions.count)
                        predictions = []
                else:
                    gesture_idx = np.argmax(res)
                
                confidence = res[gesture_idx]
                
                if confidence > confidence_threshold:
                    gesture = actions[gesture_idx]
                    if len(sentence) == 0 or sentence[-1] != gesture:
                        sentence.append(gesture)
                        if len(sentence) > 5:
                            sentence.pop(0)
                
                return {
                    "prediction": actions[gesture_idx],
                    "confidence": float(confidence),
                    "sentence": sentence.copy(),
                    "confidence_scores": res.tolist()
                }
            except Exception as e:
                print(f"Prediction error: {e}")
    
    return {
        "prediction": None,
        "confidence": 0,
        "sentence": sentence.copy(),
        "confidence_scores": [0, 0, 0]
    }

async def handle_connection(websocket, path):
    global current_mode, confidence_threshold
    
    print(f"✅ Client connected")
    
    try:
        async for message in websocket:
            try:
                # Handle JSON config messages
                if isinstance(message, str) and message.startswith('{'):
                    config = json.loads(message)
                    if "mode" in config:
                        current_mode = config["mode"]
                        print(f"Mode changed to: {current_mode}")
                    if "threshold" in config:
                        confidence_threshold = config["threshold"]
                        print(f"Threshold changed to: {confidence_threshold}")
                    continue
                
                # Handle base64 image data
                if isinstance(message, str) and message.startswith('data:image'):
                    # Extract base64 data
                    image_data = message.split(',')[1]
                    img_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    # Handle binary image data
                    nparr = np.frombuffer(message, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None and frame.size > 0:
                    result = process_frame(frame, current_mode)
                    await websocket.send(json.dumps(result))
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    # Start WebSocket server on all interfaces (for ngrok)
    async with websockets.serve(handle_connection, "0.0.0.0", 8765):
        print("=" * 50)
        print("🤟 SignSpeak Backend Server")
        print("=" * 50)
        print(f"✅ WebSocket server running on: ws://localhost:8765")
        print(f"📡 To expose to internet, run: ngrok http 8765")
        print(f"🔗 Then update WEBSOCKET_URL in index.html with your ngrok URL")
        print("=" * 50)
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
