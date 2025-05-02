import streamlit as st
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
from PIL import Image
import os
import sys
import subprocess
import time

# Install OpenCV jika belum tersedia
try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

# Load model configuration
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Load model and parameters
model = load_model('best_model.h5')
class_names = config['class_names']
img_size = tuple(config['img_size'])

# Preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, img_size)
    if config['preprocessing']['normalization'] == 'divide_by_255':
        img = img / 255.0
    return img

# Function to get camera status
def check_camera_available():
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            return False
        # Test read a frame
        ret, _ = cap.read()
        return ret
    except:
        return False
    finally:
        if cap is not None:
            cap.release()

# Function to process frame and make predictions
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
            input_img = np.expand_dims(processed_img, axis=0)

            prediction = model.predict(input_img)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            gesture = class_names[predicted_class]

            if confidence > confidence_threshold:
                # Draw bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{gesture} ({confidence:.2f})", 
                           (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 255, 0), 2)

            if show_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Streamlit app
def main():
    st.title("Hand Gesture Recognition")
    st.write("Real-time hand gesture recognition using MediaPipe and Keras")

    # Add sidebar with options
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.01)
    show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", True)
    
    # Check if camera is available
    camera_available = check_camera_available()
    
    if not camera_available:
        st.warning("""
        Camera is not accessible in this environment. 
        Using sample mode with test images instead.
        To use camera functionality, please run this app locally.
        """)
        
        # Sample mode with test images
        sample_images = {
            "Peace Sign": "sample_images/peace.jpg",
            "Thumbs Up": "sample_images/thumbs_up.jpg",
            "Open Hand": "sample_images/open_hand.jpg"
        }
        
        selected_gesture = st.selectbox("Select a sample gesture", list(sample_images.keys()))
        
        if st.button('Process Sample Image'):
            image_path = sample_images[selected_gesture]
            try:
                frame = cv2.imread(image_path)
                if frame is not None:
                    processed_frame = process_frame(frame, confidence_threshold, show_landmarks)
                    st.image(processed_frame, caption=f"Processed {selected_gesture} Sample")
                else:
                    st.error("Sample image not found. Please make sure the images are in the 'sample_images' folder.")
            except Exception as e:
                st.error(f"Error processing sample image: {str(e)}")
    else:
        # Camera mode
        run = st.checkbox('Start Camera')
        FRAME_WINDOW = st.image([])
        
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open camera")
                return
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video")
                    break
                
                processed_frame = process_frame(frame, confidence_threshold, show_landmarks)
                FRAME_WINDOW.image(processed_frame)
                time.sleep(0.1)  # Add small delay to reduce CPU usage
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            if cap is not None:
                cap.release()

if __name__ == "__main__":
    main()
