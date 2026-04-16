# app.py - Complete Sign Language to Text System
import streamlit as st
import numpy as np
from PIL import Image
import time
from datetime import datetime
import cv2
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sign Language to Text - SignSpeak",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the app - matching the reference design
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
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
        font-size: 2rem;
    }
    
    .header p {
        color: #a0a0a0;
        margin: 0;
    }
    
    /* Main content area */
    .content-area {
        background: #0f0f1a;
        border-radius: 10px;
        padding: 1rem;
        min-height: 500px;
    }
    
    /* Video/Image area */
    .video-area {
        background: #000000;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 2px solid #333;
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Text output area */
    .text-output {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        min-height: 200px;
        border: 2px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .text-output h3 {
        color: #667eea;
        margin-top: 0;
        font-size: 1rem;
    }
    
    .text-display {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        background: #0a0a0a;
        padding: 1rem;
        border-radius: 8px;
        min-height: 100px;
        font-family: monospace;
        letter-spacing: 1px;
    }
    
    /* Keyboard area */
    .keyboard-area {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .keyboard-row {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
        gap: 8px;
        flex-wrap: wrap;
    }
    
    .key-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 20px;
        margin: 5px;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
        min-width: 70px;
    }
    
    .key-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
    
    .key-btn:active {
        transform: translateY(0);
    }
    
    .action-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    .save-btn {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .space-btn {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        min-width: 300px;
    }
    
    /* Stats area */
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
    
    .stat-label {
        color: #a0a0a0;
        font-size: 0.8rem;
    }
    
    /* Detection status */
    .detection-status {
        background: #2a2a3e;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .status-active {
        color: #10b981;
        font-weight: bold;
    }
    
    .status-inactive {
        color: #ef4444;
        font-weight: bold;
    }
    
    /* Confidence bar */
    .confidence-container {
        margin-top: 10px;
    }
    
    .confidence-label {
        font-size: 0.8rem;
        color: #a0a0a0;
        margin-bottom: 5px;
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
    
    /* Recent detections */
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
        font-size: 0.9rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .key-btn {
            padding: 10px 15px;
            font-size: 1rem;
            min-width: 50px;
        }
        .space-btn {
            min-width: 200px;
        }
        .text-display {
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'text_output' not in st.session_state:
    st.session_state.text_output = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0

# Gesture mapping (A-Z)
GESTURE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Simulated gesture recognition function
def simulate_gesture_recognition(image):
    """Simulate recognizing a gesture from image"""
    # In real implementation, this would use a trained model
    # For demo, we'll simulate with some logic
    
    if image is None:
        return None, 0
    
    # Simple simulation based on image properties
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
    
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Simulate different letters based on features
    if brightness < 80:
        # Dark - likely fist (A, S)
        letter = random.choice(['A', 'S'])
        confidence = random.uniform(0.75, 0.92)
    elif brightness > 200:
        # Bright - likely open hand (B, O)
        letter = random.choice(['B', 'O'])
        confidence = random.uniform(0.70, 0.88)
    elif contrast < 30:
        # Low contrast - curved (C, G)
        letter = random.choice(['C', 'G'])
        confidence = random.uniform(0.68, 0.85)
    elif contrast > 80:
        # High contrast - V, L, Y
        letter = random.choice(['V', 'L', 'Y'])
        confidence = random.uniform(0.72, 0.90)
    else:
        # Random letter
        letter = random.choice(GESTURE_LETTERS)
        confidence = random.uniform(0.65, 0.85)
    
    return letter, confidence

def add_letter(letter):
    """Add letter to text output"""
    st.session_state.text_output += letter
    st.session_state.history.append({
        'action': 'manual',
        'value': letter,
        'time': datetime.now().strftime("%H:%M:%S")
    })

def add_space():
    """Add space to text output"""
    st.session_state.text_output += " "
    st.session_state.history.append({
        'action': 'manual',
        'value': 'space',
        'time': datetime.now().strftime("%H:%M:%S")
    })

def delete_last():
    """Delete last character"""
    if st.session_state.text_output:
        st.session_state.text_output = st.session_state.text_output[:-1]
        st.session_state.history.append({
            'action': 'delete',
            'value': 'delete',
            'time': datetime.now().strftime("%H:%M:%S")
        })

def clear_all():
    """Clear all text"""
    st.session_state.text_output = ""
    st.session_state.history.append({
        'action': 'clear',
        'value': 'cleared',
        'time': datetime.now().strftime("%H:%M:%S")
    })

def save_to_file():
    """Save text to file"""
    if st.session_state.text_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sign_text_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(st.session_state.text_output)
        return filename
    return None

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>🤟 Sign Language to Text</h1>
        <p>Convert sign language gestures to text in real-time | 26 Letters + Words</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        # Video/Camera area
        st.markdown('<div class="video-area">', unsafe_allow_html=True)
        
        # Camera input
        camera_input = st.camera_input("Show your sign language gesture", key="camera", label_visibility="collapsed")
        
        if camera_input:
            image = Image.open(camera_input)
            frame = np.array(image)
            
            # Display the image
            st.image(frame, caption="Captured Gesture", use_column_width=True)
            
            # Detection status
            detection_status = st.empty()
            
            # Simulate detection (replace with actual model)
            with st.spinner("🤟 Analyzing gesture..."):
                time.sleep(0.5)  # Simulate processing time
                letter, confidence = simulate_gesture_recognition(frame)
            
            if letter and confidence > 0.7:
                current_time = time.time() * 1000
                # Add cooldown to prevent rapid additions
                if (current_time - st.session_state.last_detection_time) > 1500:
                    st.session_state.text_output += letter
                    st.session_state.last_detection_time = current_time
                    st.session_state.current_prediction = letter
                    st.session_state.confidence = confidence
                    
                    # Add to history
                    st.session_state.history.append({
                        'action': 'detected',
                        'value': letter,
                        'confidence': confidence,
                        'time': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Success message
                    detection_status.success(f"✅ Detected: {letter} (Confidence: {confidence*100:.1f}%)")
                    st.balloons()
                else:
                    detection_status.info(f"⏳ Cooldown - Last detection: {st.session_state.current_prediction}")
            else:
                detection_status.warning("⚠️ No clear gesture detected. Please show the sign clearly.")
        else:
            st.info("📸 Click 'Browse files' to upload an image or use your camera to capture a sign language gesture")
            st.caption("Supported gestures: A-Z (American Sign Language)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detection status and confidence
        if st.session_state.current_prediction:
            st.markdown(f"""
            <div class="detection-status">
                <span>Last Detection: </span>
                <span class="status-active">{st.session_state.current_prediction}</span>
                <span> at {st.session_state.confidence*100:.1f}% confidence</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown(f"""
            <div class="confidence-container">
                <div class="confidence-label">Detection Confidence</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width: {st.session_state.confidence*100}%;">
                        {st.session_state.confidence*100:.0f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Text output area
        st.markdown("""
        <div class="text-output">
            <h3>📝 Text Output</h3>
            <div class="text-display">
        """, unsafe_allow_html=True)
        
        # Display text
        if st.session_state.text_output:
            st.markdown(f"<span style='font-size: 1.2rem;'>{st.session_state.text_output}</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #666;'>Your text will appear here...</span>", unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Action buttons
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🗑️ Clear All", use_container_width=True):
                clear_all()
                st.rerun()
        with col_b:
            if st.button("💾 Save to File", use_container_width=True):
                filename = save_to_file()
                if filename:
                    st.success(f"✅ Saved to {filename}")
                else:
                    st.warning("No text to save")
        with col_c:
            if st.button("🚪 Quit", use_container_width=True):
                st.info("Thanks for using SignSpeak!")
                st.balloons()
        
        # Statistics
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
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(st.session_state.text_output.split()) if st.session_state.text_output else 0}</div>
                <div class="stat-label">Words</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(st.session_state.history)}</div>
                <div class="stat-label">Total Actions</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Virtual Keyboard
    st.markdown("""
    <div class="keyboard-area">
        <h3 style="color: white; margin-top: 0;">🎹 Virtual Keyboard</h3>
        <p style="color: #a0a0a0; font-size: 0.9rem;">Click buttons to add letters manually or use camera for automatic detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create keyboard rows
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
                    add_letter(letter.lower() if random.random() > 0.5 else letter)
                    st.rerun()
    
    # Special buttons row
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⌫ Delete", key="delete_btn", use_container_width=True):
            delete_last()
            st.rerun()
    with col2:
        if st.button("␣ SPACE", key="space_btn", use_container_width=True):
            add_space()
            st.rerun()
    with col3:
        if st.button("🔄 Clear", key="clear_btn", use_container_width=True):
            clear_all()
            st.rerun()
    
    # Recent history
    st.markdown("""
    <div class="stats-area">
        <h3>📜 Recent Activity</h3>
        <div class="recent-list">
    """, unsafe_allow_html=True)
    
    if st.session_state.history:
        for item in st.session_state.history[-10:][::-1]:
            if item['action'] == 'detected':
                st.markdown(f"""
                <div class="recent-item">
                    <span>🎯 Detected: <b>{item['value']}</b></span>
                    <span>Confidence: {item.get('confidence', 0)*100:.1f}%</span>
                    <span style="color: #666;">{item['time']}</span>
                </div>
                """, unsafe_allow_html=True)
            elif item['action'] == 'manual':
                st.markdown(f"""
                <div class="recent-item">
                    <span>⌨️ Added: <b>{item['value']}</b></span>
                    <span style="color: #666;">{item['time']}</span>
                </div>
                """, unsafe_allow_html=True)
            elif item['action'] == 'delete':
                st.markdown(f"""
                <div class="recent-item">
                    <span>🗑️ Deleted last character</span>
                    <span style="color: #666;">{item['time']}</span>
                </div>
                """, unsafe_allow_html=True)
            elif item['action'] == 'clear':
                st.markdown(f"""
                <div class="recent-item">
                    <span>🧹 Cleared all text</span>
                    <span style="color: #666;">{item['time']}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="recent-item" style="text-align: center;">No activity yet</div>', unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 1rem; margin-top: 1rem; color: #666;">
        <p>🤟 SignSpeak - Sign Language to Text Converter | Supports A-Z Letters</p>
        <small>Use camera for automatic detection or virtual keyboard for manual input</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
