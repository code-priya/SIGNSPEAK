# app.py - REAL Sign Language Detection with Trained Model
import streamlit as st
import numpy as np
from PIL import Image
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import torch for real model
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.error("⚠️ PyTorch not installed. Please install torch to use real model.")

# Page configuration
st.set_page_config(
    page_title="Real Sign Language Detection - SignSpeak",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        border-bottom: 3px solid #667eea;
    }
    .header h1 {
        color: white;
        margin: 0;
    }
    .header p {
        color: #a0a0a0;
    }
    .video-area {
        background: #000000;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 2px solid #333;
        min-height: 400px;
    }
    .text-output {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        min-height: 250px;
        border: 2px solid #667eea;
    }
    .text-display {
        font-size: 1.3rem;
        font-weight: bold;
        color: white;
        background: #0a0a0a;
        padding: 1rem;
        border-radius: 8px;
        min-height: 120px;
        font-family: monospace;
        word-wrap: break-word;
        overflow-y: auto;
        max-height: 200px;
    }
    .keyboard-area {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .stats-area {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .stat-box {
        background: #0a0a0a;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .detection-status {
        background: #2a2a3e;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
        text-align: center;
    }
    .status-active {
        color: #10b981;
        font-weight: bold;
    }
    .confidence-bar-bg {
        background: #333;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
    }
    .confidence-bar-fill {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 100%;
        transition: width 0.3s;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.7rem;
    }
    .recent-list {
        max-height: 200px;
        overflow-y: auto;
    }
    .recent-item {
        background: #0a0a0a;
        padding: 8px;
        margin: 5px 0;
        border-radius: 5px;
        display: flex;
        justify-content: space-between;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Gesture labels (A-Y excluding J)
GESTURE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                  'V', 'W', 'X', 'Y']

# Vocabulary mapping
VOCABULARY = {
    'A': 'Apple', 'B': 'Boy', 'C': 'Cat', 'D': 'Dog', 'E': 'Elephant',
    'F': 'Friend', 'G': 'Good', 'H': 'Hello', 'I': 'I', 'K': 'Kite',
    'L': 'Love', 'M': 'Mother', 'N': 'No', 'O': 'Open', 'P': 'Please',
    'Q': 'Question', 'R': 'Run', 'S': 'Sorry', 'T': 'Thank', 'U': 'You',
    'V': 'Very', 'W': 'Welcome', 'X': 'X-ray', 'Y': 'Yes'
}

# REAL CNN Model Architecture (same as training)
if TORCH_AVAILABLE:
    class SignLanguageCNN(nn.Module):
        def __init__(self, num_classes=24):
            super(SignLanguageCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Linear(256 * 1 * 1, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_classes)
            
        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            x = x.view(x.size(0), -1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = self.fc3(x)
            return x

# Initialize session state
if 'text_output' not in st.session_state:
    st.session_state.text_output = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_real_model():
    """Load the REAL trained PyTorch model"""
    if not TORCH_AVAILABLE:
        st.error("❌ PyTorch not available. Cannot load model.")
        return None
    
    model = SignLanguageCNN(num_classes=24)
    
    try:
        # Try to load the trained model
        model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        model.eval()
        st.success("✅ REAL model loaded successfully! Ready for sign language detection.")
        return model
    except FileNotFoundError:
        st.error("❌ model.pth not found! Please train the model first using train_model.py")
        st.info("📝 Running in DEMO mode. Train model to get real predictions.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def preprocess_image_for_model(image):
    """REAL preprocessing - must match training exactly"""
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image
        
        # Resize to 28x28 (same as training)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(gray.astype('uint8'))
        resized = pil_img.resize((28, 28))
        
        # Convert to array and normalize to 0-1
        img_array = np.array(resized).astype('float32') / 255.0
        
        # Invert if needed (MNIST style: white on black)
        # For hand signs, we want black background with white hand
        if np.mean(img_array) > 0.5:
            img_array = 1.0 - img_array
        
        # Add batch and channel dimensions
        tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None

def predict_real_gesture(model, image, confidence_threshold=0.7):
    """REAL prediction using trained model"""
    if model is None:
        # Fallback to demo mode if no model
        return None, 0, None, None
    
    try:
        # Preprocess image
        tensor = preprocess_image_for_model(image)
        if tensor is None:
            return None, 0, None, None
        
        # Make prediction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        confidence = confidence.item()
        gesture_idx = predicted.item()
        
        if confidence > confidence_threshold and gesture_idx < len(GESTURE_LABELS):
            gesture = GESTURE_LABELS[gesture_idx]
            word = VOCABULARY.get(gesture, gesture)
            return word, confidence, gesture, probabilities[0].tolist()
        
        return None, confidence, None, probabilities[0].tolist()
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0, None, None

def add_letter(letter):
    st.session_state.text_output += letter
    st.session_state.history.append({
        'action': 'manual',
        'value': letter,
        'time': datetime.now().strftime("%H:%M:%S")
    })

def add_space():
    st.session_state.text_output += " "
    st.session_state.history.append({
        'action': 'manual',
        'value': 'space',
        'time': datetime.now().strftime("%H:%M:%S")
    })

def delete_last():
    if st.session_state.text_output:
        st.session_state.text_output = st.session_state.text_output[:-1]
        st.session_state.history.append({
            'action': 'delete',
            'value': 'delete',
            'time': datetime.now().strftime("%H:%M:%S")
        })

def clear_all():
    st.session_state.text_output = ""
    st.session_state.history = []
    st.session_state.total_predictions = 0
    st.session_state.history.append({
        'action': 'clear',
        'value': 'cleared',
        'time': datetime.now().strftime("%H:%M:%S")
    })

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>🤟 REAL Sign Language Detection</h1>
        <p>Using Trained CNN Model on Sign Language MNIST Dataset | 24 Letters (A-Y)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load REAL model
    model = load_real_model()
    
    if model is None:
        st.warning("⚠️ Running in DEMO mode. Train model using train_model.py for real predictions.")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown('<div class="video-area">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: white;'>📸 Capture Sign Language</h3>", unsafe_allow_html=True)
        
        camera_input = st.camera_input("Show your sign language gesture", key="camera", label_visibility="collapsed")
        
        if camera_input:
            image = Image.open(camera_input)
            st.image(image, caption="Captured Gesture", use_column_width=True)
            
            detection_status = st.empty()
            
            with st.spinner("🤟 Running REAL model prediction..."):
                time.sleep(0.2)  # Small delay for UX
                word, confidence, gesture, all_probs = predict_real_gesture(model, image, confidence_threshold=0.7)
            
            if word:
                current_time = time.time() * 1000
                if (current_time - st.session_state.last_detection_time) > 1500:
                    st.session_state.text_output += word
                    st.session_state.last_detection_time = current_time
                    st.session_state.total_predictions += 1
                    
                    st.session_state.history.append({
                        'action': 'detected',
                        'value': word,
                        'gesture': gesture,
                        'confidence': confidence,
                        'time': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    detection_status.success(f"✅ REAL Detection: {word} (Letter: {gesture}) | Confidence: {confidence*100:.1f}%")
                    st.balloons()
                else:
                    detection_status.info(f"⏳ Cooldown - Last detection: {word}")
            else:
                detection_status.warning(f"⚠️ No clear gesture detected (Confidence: {confidence*100:.1f}%)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="text-output">
            <h3>📝 Text Output</h3>
            <div class="text-display">
        """, unsafe_allow_html=True)
        
        if st.session_state.text_output:
            st.markdown(f"{st.session_state.text_output}")
        else:
            st.markdown("<span style='color: #666;'>Your text will appear here...</span>", unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🗑️ Clear All", use_container_width=True):
                clear_all()
                st.rerun()
        with col_b:
            if st.button("💾 Save to File", use_container_width=True):
                if st.session_state.text_output:
                    st.download_button(
                        label="📥 Download",
                        data=st.session_state.text_output,
                        file_name=f"sign_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No text to save")
        with col_c:
            if st.button("🔄 Reset", use_container_width=True):
                clear_all()
                st.rerun()
        
        st.markdown("""
        <div class="stats-area">
            <h3>📊 Statistics</h3>
        """, unsafe_allow_html=True)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(st.session_state.text_output)}</div>
                <div class="stat-label">Characters</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s2:
            word_count = len(st.session_state.text_output.split()) if st.session_state.text_output else 0
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{word_count}</div>
                <div class="stat-label">Words</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{st.session_state.total_predictions}</div>
                <div class="stat-label">Detections</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Virtual Keyboard
    st.markdown("""
    <div class="keyboard-area">
        <h3>🎹 Virtual Keyboard (Manual Input)</h3>
        <p style="color: #a0a0a0;">Use camera for real detection OR keyboard for manual input</p>
    </div>
    """, unsafe_allow_html=True)
    
    rows = [
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
    ]
    
    for row in rows:
        cols = st.columns(len(row))
        for i, letter in enumerate(row):
            with cols[i]:
                if st.button(letter, key=f"key_{letter}", use_container_width=True):
                    add_letter(letter.lower())
                    st.rerun()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⌫ Delete", use_container_width=True):
            delete_last()
            st.rerun()
    with col2:
        if st.button("␣ SPACE", use_container_width=True):
            add_space()
            st.rerun()
    with col3:
        if st.button("🧹 Clear", use_container_width=True):
            clear_all()
            st.rerun()
    
    # Recent history
    st.markdown("""
    <div class="stats-area">
        <h3>📜 Recent Detections</h3>
        <div class="recent-list">
    """, unsafe_allow_html=True)
    
    if st.session_state.history:
        for item in st.session_state.history[-10:][::-1]:
            if item['action'] == 'detected':
                st.markdown(f"""
                <div class="recent-item">
                    <span>🎯 {item['value']} ({item.get('gesture', '?')})</span>
                    <span>{item['confidence']*100:.1f}%</span>
                    <span>{item['time']}</span>
                </div>
                """, unsafe_allow_html=True)
            elif item['action'] == 'manual':
                st.markdown(f"""
                <div class="recent-item">
                    <span>⌨️ Added: {item['value']}</span>
                    <span>{item['time']}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="recent-item">No detections yet</div>', unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Footer with model info
    if model:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; margin-top: 1rem; color: #10b981;">
            <p>✅ REAL MODEL ACTIVE - Trained on Sign Language MNIST (24 letters)</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; margin-top: 1rem; color: #f59e0b;">
            <p>⚠️ DEMO MODE - Train model using train_model.py to enable real detection</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
