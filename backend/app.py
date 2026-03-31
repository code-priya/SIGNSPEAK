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

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load trained model
model = tf.keras.models.load_model('models/model_weights.h5', compile=False)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Gesture mapping
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Detection parameters
sequence_length = 30
sequence = deque(maxlen=sequence_length)
sentence = []
predictions = []
threshold = 0.7

# Mode settings
current_mode = "standard"  # standard, advanced, batch
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
            res = model.predict(np.expand_dims(np.array(sequence), axis=0), verbose=0)[0]
            
            if mode == "advanced":
                # Smooth predictions with moving average
                predictions.append(np.argmax(res))
                if len(predictions) > 10:
                    predictions.pop(0)
                gesture_idx = max(set(predictions), key=predictions.count) if predictions else np.argmax(res)
            elif mode == "batch":
                # Batch mode - only update every 10 frames
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
    
    return {
        "prediction": None,
        "confidence": 0,
        "sentence": sentence.copy(),
        "confidence_scores": [0, 0, 0]
    }

async def handle_connection(websocket, path):
    global current_mode, confidence_threshold
    
    print(f"Client connected from {websocket.remote_address}")
    
    try:
        async for message in websocket:
            try:
                # Check if message is JSON config or image
                if isinstance(message, str):
                    config = json.loads(message)
                    if "mode" in config:
                        current_mode = config["mode"]
                    if "threshold" in config:
                        confidence_threshold = config["threshold"]
                    continue
                
                # Process image frame
                nparr = np.frombuffer(message, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    result = process_frame(frame, current_mode)
                    await websocket.send(json.dumps(result))
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected from {websocket.remote_address}")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
