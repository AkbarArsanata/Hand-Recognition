import streamlit as st
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
from PIL import Image
import os
import cv2

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

# Load model and config
try:
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    model = load_model('best_model.h5')
    model_loaded = True
except Exception as e:
    st.error(f"Model loading error: {str(e)}")
    model_loaded = False
    config = {
        'class_names': ['palm', 'J', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down'],
        'img_size': [224, 224],
        'preprocessing': {'normalization': 'divide_by_255'}
    }

class_names = config['class_names']
img_size = tuple(config['img_size'])

# Gesture guide with example images (replace with your actual image paths)
GESTURE_GUIDE = {
    "01_palm": {
        "name": "Palm",
        "description": "Hand fully open with fingers extended",
        "example": "gesture_examples/palm.jpg"
    },
    "02_J": {
        "name": "J Sign",
        "description": "Thumb and index finger forming a 'J' shape",
        "example": "gesture_examples/J.jpg"
    },
    # Add all other gestures similarly
    "10_down": {
        "name": "Thumbs Down",
        "description": "Thumb pointing downward",
        "example": "gesture_examples/down.jpg"
    }
}

def show_gesture_guide():
    st.header("Gesture Guide")
    st.write("Here are the gestures this app can recognize:")
    
    cols = st.columns(3)
    for i, (gesture_id, info) in enumerate(GESTURE_GUIDE.items()):
        with cols[i % 3]:
            try:
                img = Image.open(info["example"])
                st.image(img, caption=f"{info['name']}: {info['description']}", width=200)
            except:
                st.warning(f"Example image not found for {info['name']}")
                st.write(f"**{info['name']}**: {info['description']}")

def preprocess_image(img):
    img = cv2.resize(img, img_size)
    if config['preprocessing']['normalization'] == 'divide_by_255':
        img = img / 255.0
    return img

def process_frame(frame, confidence_threshold, show_landmarks):
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
            
            if model_loaded:
                input_img = np.expand_dims(processed_img, axis=0)
                prediction = model.predict(input_img, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                gesture = class_names[predicted_class]
            else:
                gesture = "demo_gesture"
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
    st.title("👋 Hand Gesture Recognition")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Recognizer", "Gesture Guide", "About"])
    
    with tab1:
        st.header("Real-Time Gesture Recognition")
        
        with st.sidebar:
            st.subheader("Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold", 0.0, 1.0, 0.7, 0.01,
                help="Minimum confidence level to display recognition"
            )
            show_landmarks = st.checkbox(
                "Show Hand Landmarks", True,
                help="Display MediaPipe hand landmarks"
            )
            
            if not model_loaded:
                st.warning("Running in demo mode (model not fully loaded)")
        
        input_mode = st.radio(
            "Input Mode:",
            ("Live Camera", "Upload Image"),
            horizontal=True
        )
        
        if input_mode == "Live Camera":
            img_file_buffer = st.camera_input("Show your hand to the camera")
            if img_file_buffer is not None:
                image = Image.open(img_file_buffer)
                frame = np.array(image)
                processed_frame = process_frame(frame, confidence_threshold, show_landmarks)
                st.image(processed_frame, caption="Processed Result", use_column_width=True)
        else:
            uploaded_file = st.file_uploader("Upload a hand gesture image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                frame = np.array(image)
                processed_frame = process_frame(frame, confidence_threshold, show_landmarks)
                st.image(processed_frame, caption="Processed Result", use_column_width=True)
    
    with tab2:
        show_gesture_guide()
    
    with tab3:
        st.header("About This App")
        st.markdown("""
        This app recognizes hand gestures in real-time using:
        - MediaPipe for hand tracking
        - Keras/TensorFlow for gesture classification
        
        **Supported Gestures:**
        - Palm (Open Hand)
        - J Sign
        - Fist
        - Thumbs Up/Down
        - OK Sign
        - And more!
        """)

if __name__ == "__main__":
    main()
