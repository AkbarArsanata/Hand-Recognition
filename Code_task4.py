import streamlit as st
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
from PIL import Image
import os
import cv2
import urllib.request
import tempfile

# Set page config
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="👋",
    layout="wide"
)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Sample model configuration (fallback if file not found)
DEFAULT_CONFIG = {
    'class_names': ['Open', 'Closed', 'Pointing', 'Thumbs Up', 'Peace'],
    'img_size': [224, 224],
    'preprocessing': {
        'normalization': 'divide_by_255'
    }
}

@st.cache_resource
def load_model_with_fallback():
    """Load model with fallback to sample data if files not found"""
    try:
        # Try to download model files if they don't exist
        if not os.path.exists('model_config.json'):
            try:
                urllib.request.urlretrieve(
                    'https://github.com/yourusername/yourrepo/raw/main/model_config.json',
                    'model_config.json'
                )
            except:
                st.warning("Could not download model config, using default")
                with open('model_config.json', 'w') as f:
                    json.dump(DEFAULT_CONFIG, f)
        
        if not os.path.exists('best_model.h5'):
            try:
                with st.spinner("Downloading model (this may take a minute)..."):
                    urllib.request.urlretrieve(
                        'https://github.com/yourusername/yourrepo/raw/main/best_model.h5',
                        'best_model.h5'
                    )
            except Exception as e:
                st.error(f"Could not download model: {str(e)}")
                return None, DEFAULT_CONFIG
        
        # Load the files
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        model = load_model('best_model.h5')
        return model, config
    
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, DEFAULT_CONFIG

# Load model and config
model, config = load_model_with_fallback()
class_names = config['class_names']
img_size = tuple(config['img_size'])

def preprocess_image(img):
    """Preprocess image according to model requirements"""
    img = cv2.resize(img, img_size)
    if config['preprocessing']['normalization'] == 'divide_by_255':
        img = img / 255.0
    return img

def process_frame(frame, confidence_threshold, show_landmarks):
    """Process a single frame for hand gesture recognition"""
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            padding = 20
            x_min = max(0, int(x_min) - padding)
            y_min = max(0, int(y_min) - padding)
            x_max = min(w, int(x_max) + padding)
            y_max = min(h, int(y_max) + padding)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size == 0:
                continue

            processed_img = preprocess_image(hand_img)
            
            if model is not None:
                input_img = np.expand_dims(processed_img, axis=0)
                prediction = model.predict(input_img, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                gesture = class_names[predicted_class]
            else:
                # Demo mode with mock predictions
                gesture = "Demo Gesture"
                confidence = 0.85

            if confidence > confidence_threshold:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{gesture} ({confidence:.2f})", 
                           (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 255, 0), 2)

            if show_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def main():
    st.title("👋 Real-Time Hand Gesture Recognition")
    st.markdown("""
    This app recognizes hand gestures in real-time using:
    - MediaPipe for hand tracking
    - Keras/TensorFlow for gesture classification
    """)
    
    with st.sidebar:
        st.header("Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.7, 0.01,
            help="Minimum confidence level to display recognition"
        )
        show_landmarks = st.checkbox(
            "Show Hand Landmarks", 
            True,
            help="Display MediaPipe hand landmarks"
        )
        
        if model is None:
            st.warning("Running in demo mode (model not loaded)")
            st.info("To use the full model, ensure 'best_model.h5' and 'model_config.json' are available")
    
    tab1, tab2 = st.tabs(["Live Camera", "Upload Image"])
    
    with tab1:
        st.subheader("Live Camera Recognition")
        img_file_buffer = st.camera_input(
            "Show your hand to the camera",
            help="The app will try to recognize your hand gesture"
        )
        
        if img_file_buffer is not None:
            try:
                image = Image.open(img_file_buffer)
                frame = np.array(image)
                processed_frame = process_frame(frame, confidence_threshold, show_landmarks)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_column_width=True)
                with col2:
                    st.image(processed_frame, caption="Processed", use_column_width=True)
                    
            except Exception as e:
                st.error(f"Error processing frame: {str(e)}")
    
    with tab2:
        st.subheader("Image Upload Recognition")
        uploaded_file = st.file_uploader(
            "Upload a hand gesture image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image containing a hand gesture"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                frame = np.array(image)
                processed_frame = process_frame(frame, confidence_threshold, show_landmarks)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_column_width=True)
                with col2:
                    st.image(processed_frame, caption="Processed", use_column_width=True)
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
