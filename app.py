import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import os

# Try importing mediapipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Sign Language to Text",
    page_icon="🤟",
    layout="centered"
)

# Custom CSS for minimal styling
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-letter {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
    }
    .confidence-score {
        font-size: 24px;
        color: #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

# CNN Model (same as in train.py)
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=24):
        super(SignLanguageCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

@st.cache_resource
def load_model():
    """Load the trained model"""
    if not os.path.exists('model.pth'):
        st.error("⚠️ Model file 'model.pth' not found. Please run train.py first!")
        return None, None, None, None
    
    try:
        checkpoint = torch.load('model.pth', map_location='cpu')
        
        # Get model parameters
        classes = checkpoint['classes']
        img_size = checkpoint.get('img_size', 64)
        transform_params = checkpoint.get('transform_params', {'mean': [0.5], 'std': [0.5]})
        
        # Initialize model
        model = SignLanguageCNN(num_classes=len(classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, classes, img_size, transform_params
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def preprocess_image(image, img_size, transform_params):
    """Preprocess image exactly like training"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_params['mean'], std=transform_params['std'])
    ])
    
    return transform(image).unsqueeze(0)

def detect_hand_mediapipe(image):
    """Detect and crop hand region using MediaPipe"""
    if not MEDIAPIPE_AVAILABLE:
        return image
    
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = hands.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            h, w, _ = image_cv.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
            
            # Add padding
            padding = 30
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Crop hand region
            if x_max > x_min and y_max > y_min:
                image_cv = image_cv[y_min:y_max, x_min:x_max]
                return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    except Exception as e:
        st.warning(f"Hand detection failed: {str(e)}")
    
    return image

def main():
    st.title("🤟 Sign Language to Text")
    st.markdown("Upload an image or use your camera to translate sign language letters")
    
    # Load model
    model_data = load_model()
    if model_data is None:
        st.info("👆 Please run train.py first to train the model")
        return
    
    model, classes, img_size, transform_params = model_data
    st.success(f"✅ Model loaded! Recognizes {len(classes)} letters")
    
    # Input selection
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload Image", "📸 Camera Input"],
        horizontal=True
    )
    
    image = None
    
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
    else:
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            image = Image.open(camera_image)
    
    # Process image
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)
        
        # Hand detection option
        use_hand_detection = st.checkbox("Use hand detection (MediaPipe)", value=True)
        
        processed_image = image.copy()
        if use_hand_detection and MEDIAPIPE_AVAILABLE:
            with st.spinner("Detecting hand..."):
                processed_image = detect_hand_mediapipe(image)
            
            with col2:
                st.markdown("**Detected Hand**")
                st.image(processed_image, use_container_width=True)
        
        # Predict button
        if st.button("🔍 Predict", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    # Preprocess image
                    img_tensor = preprocess_image(processed_image, img_size, transform_params)
                    
                    # Make prediction
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        predicted_class = classes[predicted.item()]
                        confidence_score = confidence.item()
                    
                    # Display results
                    st.markdown("---")
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'<p class="prediction-letter">{predicted_class}</p>', 
                                  unsafe_allow_html=True)
                        st.markdown("**Predicted Letter**")
                    
                    with col2:
                        st.markdown(f'<p class="confidence-score">{confidence_score:.2%}</p>', 
                                  unsafe_allow_html=True)
                        st.markdown("**Confidence Score**")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show top 3 predictions
                    st.markdown("**Top 3 Predictions:**")
                    top_probs, top_indices = torch.topk(probabilities, min(3, len(classes)))
                    
                    for i in range(len(top_indices[0])):
                        letter = classes[top_indices[0][i].item()]
                        prob = top_probs[0][i].item()
                        st.progress(prob, text=f"{letter}: {prob:.2%}")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        
        # Clear button
        if st.button("🗑️ Clear"):
            st.rerun()
    else:
        st.info("👆 Please upload an image or take a picture to start")
    
    # Footer
    st.markdown("---")
    st.markdown("*Made with ❤️ using PyTorch and Streamlit*")

if __name__ == "__main__":
    main()
