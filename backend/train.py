import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, 
    Flatten, BatchNormalization, Bidirectional, Attention,
    GlobalAveragePooling1D, Concatenate, Reshape
)
from tensorflow.keras.callbacks import (
    TensorBoard, EarlyStopping, ReduceLROnPlateau, 
    ModelCheckpoint, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import json
from datetime import datetime
import albumentations as A
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define gestures (expandable - add more gestures here)
ACTIONS = np.array([
    'hello', 'thanks', 'iloveyou', 'yes', 'no', 
    'please', 'sorry', 'help', 'eat', 'drink',
    'sleep', 'good', 'bad', 'happy', 'sad'
])

SEQUENCE_LENGTH = 30
FEATURES_DIM = 1662  # MediaPipe features dimension

class DataCollector:
    """Enhanced data collection with augmentation"""
    
    def __init__(self, data_path='data/MP_Data'):
        self.data_path = data_path
        self.actions = ACTIONS
        self.sequence_length = SEQUENCE_LENGTH
        
    def setup_directories(self):
        """Create directory structure for data collection"""
        os.makedirs(self.data_path, exist_ok=True)
        for action in self.actions:
            action_path = os.path.join(self.data_path, action)
            os.makedirs(action_path, exist_ok=True)
            for sequence in range(30):  # 30 sequences per gesture
                sequence_path = os.path.join(action_path, str(sequence))
                os.makedirs(sequence_path, exist_ok=True)
        print(f"✅ Directory structure created at {self.data_path}")
    
    def collect_landmarks(self, num_sequences=30, sequence_length=30):
        """Interactive data collection"""
        cap = cv2.VideoCapture(0)
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            for action in self.actions:
                print(f"\n📹 Preparing to collect data for: {action}")
                input("Press ENTER when ready...")
                
                for sequence in range(num_sequences):
                    print(f"  Sequence {sequence + 1}/{num_sequences}")
                    
                    # Countdown
                    for frame_num in range(sequence_length):
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        
                        # Show countdown
                        cv2.putText(frame, f'Recording: {action} - Seq {sequence + 1}', 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f'Frame: {frame_num + 1}/{sequence_length}', 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Process frame
                        image, results = self._mediapipe_detection(frame, holistic)
                        self._draw_styled_landmarks(image, results)
                        
                        # Extract and save keypoints
                        keypoints = self._extract_keypoints(results)
                        npy_path = os.path.join(self.data_path, action, str(sequence), f"{frame_num}.npy")
                        np.save(npy_path, keypoints)
                        
                        cv2.imshow('Data Collection', image)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                    
                    print(f"    ✅ Completed sequence {sequence + 1}")
            
            cap.release()
            cv2.destroyAllWindows()
            print("\n🎉 Data collection complete!")
    
    def _mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def _draw_styled_landmarks(self, image, results):
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1))
        # Draw hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1))
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1))
    
    def _extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

class DataAugmentor:
    """Data augmentation for training"""
    
    @staticmethod
    def add_noise(sequence, noise_factor=0.01):
        noise = np.random.randn(*sequence.shape) * noise_factor
        return sequence + noise
    
    @staticmethod
    def time_shift(sequence, shift_range=5):
        shift = np.random.randint(-shift_range, shift_range)
        return np.roll(sequence, shift, axis=0)
    
    @staticmethod
    def scale(sequence, scale_range=(0.9, 1.1)):
        scale_factor = np.random.uniform(*scale_range)
        return sequence * scale_factor
    
    @staticmethod
    def augment_sequence(sequence):
        """Apply random augmentation"""
        aug_type = np.random.choice(['noise', 'shift', 'scale', 'none'])
        if aug_type == 'noise':
            return DataAugmentor.add_noise(sequence)
        elif aug_type == 'shift':
            return DataAugmentor.time_shift(sequence)
        elif aug_type == 'scale':
            return DataAugmentor.scale(sequence)
        return sequence

class CNNLSTMModel:
    """Hybrid CNN + LSTM model for sign language recognition"""
    
    def __init__(self, input_shape=(SEQUENCE_LENGTH, FEATURES_DIM), num_classes=len(ACTIONS)):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build CNN + LSTM hybrid architecture"""
        
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # CNN branch for spatial feature extraction
        cnn_branch = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        
        cnn_branch = Conv1D(128, kernel_size=3, padding='same', activation='relu')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        
        cnn_branch = Conv1D(256, kernel_size=3, padding='same', activation='relu')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = GlobalAveragePooling1D()(cnn_branch)
        
        # LSTM branch for temporal features
        lstm_branch = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(inputs)
        lstm_branch = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3))(lstm_branch)
        
        # Attention mechanism
        attention = Attention()([lstm_branch, lstm_branch])
        
        # Combine branches
        combined = Concatenate()([cnn_branch, lstm_branch, attention])
        
        # Dense layers
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with AdamW optimizer
        optimizer = Adam(learning_rate=0.0005, weight_decay=0.0001)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy', 'accuracy']
        )
        
        return self.model
    
    def load_data(self, data_path='data/MP_Data', augment=True):
        """Load and preprocess data"""
        sequences, labels = [], []
        
        print("📂 Loading data...")
        for action_idx, action in enumerate(tqdm(ACTIONS)):
            action_path = os.path.join(data_path, action)
            if not os.path.exists(action_path):
                print(f"⚠️ Warning: {action_path} not found. Skipping {action}")
                continue
                
            for sequence in os.listdir(action_path):
                sequence_path = os.path.join(action_path, sequence)
                window = []
                
                for frame_num in range(SEQUENCE_LENGTH):
                    frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                    if os.path.exists(frame_path):
                        res = np.load(frame_path)
                        window.append(res)
                    else:
                        # Handle missing frames
                        window.append(np.zeros(FEATURES_DIM))
                
                if len(window) == SEQUENCE_LENGTH:
                    window = np.array(window)
                    sequences.append(window)
                    labels.append(action_idx)
                    
                    # Add augmented data
                    if augment:
                        aug_window = DataAugmentor.augment_sequence(window)
                        sequences.append(aug_window)
                        labels.append(action_idx)
        
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        
        print(f"✅ Loaded {len(X)} samples with {self.num_classes} classes")
        return train_test_split(X, y, test_size=0.15, random_state=42, stratify=labels)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
        """Train the model"""
        
        # Callbacks
        callbacks = [
            TensorBoard(log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'),
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
            ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
            CSVLogger('training_log.csv', append=True)
        ]
        
        print("\n🚀 Starting training...")
        print(f"Training samples: {X_train.shape}")
        print(f"Validation samples: {X_val.shape}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def save_model(self, path='models'):
        """Save model and metadata"""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(path, 'cnn_lstm_model.h5'))
        self.model.save_weights(os.path.join(path, 'model_weights.h5'))
        
        # Save metadata
        metadata = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'actions': ACTIONS.tolist(),
            'sequence_length': SEQUENCE_LENGTH,
            'features_dim': FEATURES_DIM
        }
        
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved to {path}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        loss, accuracy, cat_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n📊 Test Results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Categorical Accuracy: {cat_acc:.4f}")
        return loss, accuracy

class GesturePredictor:
    """Real-time gesture predictor using trained model"""
    
    def __init__(self, model_path='models/cnn_lstm_model.h5'):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.sequence_length = SEQUENCE_LENGTH
        
        # Load metadata
        with open('models/metadata.json', 'r') as f:
            self.metadata = json.load(f)
        self.actions = np.array(self.metadata['actions'])
        
    def predict(self, keypoints, confidence_threshold=0.7):
        """Predict gesture from keypoints"""
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-self.sequence_length:]
        
        if len(self.sequence) == self.sequence_length:
            res = self.model.predict(np.expand_dims(np.array(self.sequence), axis=0), verbose=0)[0]
            gesture_idx = np.argmax(res)
            confidence = res[gesture_idx]
            
            if confidence > confidence_threshold:
                gesture = self.actions[gesture_idx]
                if len(self.sentence) == 0 or self.sentence[-1] != gesture:
                    self.sentence.append(gesture)
                    if len(self.sentence) > 5:
                        self.sentence.pop(0)
                
                return {
                    'gesture': gesture,
                    'confidence': float(confidence),
                    'sentence': self.sentence.copy(),
                    'all_scores': res.tolist()
                }
        
        return None

def plot_training_history(history):
    """Plot training metrics"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("🤟 Sign Language Recognition - CNN+LSTM Hybrid Model")
    print("=" * 60)
    
    # Step 1: Collect data
    print("\n📹 Step 1: Data Collection")
    print(f"Supported gestures: {', '.join(ACTIONS[:10])}...")
    collector = DataCollector()
    
    choice = input("\nDo you want to collect new data? (y/n): ")
    if choice.lower() == 'y':
        collector.setup_directories()
        collector.collect_landmarks(num_sequences=30)
    
    # Step 2: Build and train model
    print("\n🧠 Step 2: Building CNN+LSTM Model")
    model = CNNLSTMModel(num_classes=len(ACTIONS))
    model.build_model()
    model.model.summary()
    
    # Load data
    X_train, X_val, y_train, y_val = model.load_data(augment=True)
    
    # Train
    history = model.train(X_train, y_train, X_val, y_val, epochs=150)
    
    # Evaluate
    X_train, X_test, y_train, y_test = model.load_data(augment=False)
    model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model()
    
    # Plot results
    plot_training_history(history)
    
    print("\n🎉 Training complete! Model saved to models/")
