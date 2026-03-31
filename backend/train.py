import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import time

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

def collect_data():
    DATA_PATH = os.path.join('..', 'data', 'MP_Data')
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    no_sequences = 30
    sequence_length = 30
    
    os.makedirs(DATA_PATH, exist_ok=True)
    
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        os.makedirs(action_path, exist_ok=True)
        for sequence in range(no_sequences):
            sequence_path = os.path.join(action_path, str(sequence))
            os.makedirs(sequence_path, exist_ok=True)
    
    print("Data collection complete. Directory structure created.")

def load_data():
    DATA_PATH = os.path.join('..', 'data', 'MP_Data')
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    sequence_length = 30
    
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}
    
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        for sequence in np.array(os.listdir(action_path)).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(action_path, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    return train_test_split(X, y, test_size=0.05, random_state=42)

def create_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model

def train_model():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training samples: {X_train.shape}")
    print(f"Testing samples: {X_test.shape}")
    
    print("Creating model...")
    model = create_model((30, 1662), 3)
    model.summary()
    
    callbacks = [
        TensorBoard(log_dir='logs'),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
    ]
    
    print("Training model for 150 epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=150,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_weights('models/model_weights.h5')
    model.save('models/model_weights.h5')
    print("Model saved to models/model_weights.h5")
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return model, history

if __name__ == "__main__":
    print("=== SignSpeak Training Script ===")
    print("1. Collect data (create directories)")
    print("2. Train model")
    
    collect_data()
    print("\nPlease add training data to the MP_Data directory before training.")
    print("Run the data collection script first to record gestures.\n")
    
    train_model()
