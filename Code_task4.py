import streamlit as st
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
from PIL import Image
import os
import cv2
import time
import pandas as pd
import plotly.express as px

# Set page config with author information
st.set_page_config(
    page_title="GestureSense Pro | Hand Gesture Recognition",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        color: #4F8BF9;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .subheader {
        color: #2C3E50;
        font-size: 1.3em;
        margin-bottom: 1em;
    }
    .feature-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f0f2f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        padding: 10px;
        margin-top: 30px;
        border-top: 1px solid #e1e4e8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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
        'class_names': ['palm', 'fist', 'point', 'ok', 'peace', 'rock', 'thumbs_up', 'thumbs_down'],
        'img_size': [224, 224],
        'preprocessing': {'normalization': 'divide_by_255'}
    }

class_names = config['class_names']
img_size = tuple(config['img_size'])

# Gesture guide with emojis
GESTURE_GUIDE = {
    "palm": {"emoji": "✋", "desc": "Open hand with fingers extended"},
    "fist": {"emoji": "✊", "desc": "Closed fist"},
    "point": {"emoji": "👉", "desc": "Index finger pointing"},
    "ok": {"emoji": "👌", "desc": "Thumb and index finger making a circle"},
    "peace": {"emoji": "✌️", "desc": "Index and middle finger raised"},
    "rock": {"emoji": "🤘", "desc": "Index and pinky finger raised"},
    "thumbs_up": {"emoji": "👍", "desc": "Thumb raised upward"},
    "thumbs_down": {"emoji": "👎", "desc": "Thumb pointing downward"}
}

# Performance metrics
if 'performance_data' not in st.session_state:
    st.session_state.performance_data = {
        "timestamp": [],
        "gesture": [],
        "confidence": [],
        "processing_time": []
    }

# Preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, img_size)
    if config['preprocessing']['normalization'] == 'divide_by_255':
        img = img / 255.0
    return img

# Frame processing with performance tracking
def process_frame(frame, confidence_threshold):
    start_time = time.time()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    gesture = None
    confidence = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand bounding box
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
                
                # Store performance data
                st.session_state.performance_data["timestamp"].append(time.time())
                st.session_state.performance_data["gesture"].append(gesture)
                st.session_state.performance_data["confidence"].append(confidence)
                st.session_state.performance_data["processing_time"].append(time.time() - start_time)

            # Draw landmarks and bounding box
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            if gesture and confidence > confidence_threshold:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{GESTURE_GUIDE.get(gesture, {}).get('emoji', '')} {gesture} ({confidence:.2f})", 
                          (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9, (0, 255, 0), 2)
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def show_author_info():
    st.markdown("""
    <div class="footer">
        <p>Developed by <strong>Ibrahim Akbar Arsanata</strong> | 
        <a href="https://www.linkedin.com/in/ibrahim-akbar-arsanata" target="_blank">LinkedIn</a> | 
        <a href="mailto:arsanataibrahim9@gmail.com">Email</a></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<p class="header">GestureSense Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Advanced Hand Gesture Recognition System</p>', unsafe_allow_html=True)
    
    # Sidebar with author info and settings
    with st.sidebar:
        st.markdown("""
        ### About the Developer
        **Ibrahim Akbar Arsanata**  
        Computer Vision Specialist  
        [LinkedIn Profile](https://www.linkedin.com/in/ibrahim-akbar-arsanata)  
        arsanataibrahim9@gmail.com
        """)
        
        st.markdown("---")
        st.subheader("Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.7, 0.01,
            help="Minimum confidence level to display recognition"
        )
        show_analytics = st.checkbox("Show Performance Analytics", True)
        st.markdown("---")
        
        if not model_loaded:
            st.warning("Running in demo mode (model not fully loaded)")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Real-Time", "Gesture Guide", "Analytics"])
    
    with tab1:
        st.subheader("Real-Time Gesture Recognition")
        
        # Use regular camera input instead of WebRTC
        img_file_buffer = st.camera_input("Show your hand to the camera")
        
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            frame = np.array(image)
            processed_frame = process_frame(frame, confidence_threshold)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.image(processed_frame, caption="Processed Frame", use_column_width=True)
            
            with col2:
                if st.session_state.performance_data["gesture"]:
                    last_gesture = st.session_state.performance_data["gesture"][-1]
                    last_confidence = st.session_state.performance_data["confidence"][-1]
                    st.markdown(f"**Gesture**: {GESTURE_GUIDE.get(last_gesture, {}).get('emoji', '')} {last_gesture}")
                    st.markdown(f"**Confidence**: {last_confidence:.2%}")
    
    with tab2:
        st.subheader("Gesture Guide")
        st.markdown("Learn how to perform the supported gestures:")
        
        cols = st.columns(4)
        for i, (gesture, info) in enumerate(GESTURE_GUIDE.items()):
            with cols[i % 4]:
                with st.container():
                    st.markdown(f'<div class="feature-card">'
                               f'<h3>{info["emoji"]} {gesture.replace("_", " ").title()}</h3>'
                               f'<p>{info["desc"]}</p>'
                               f'</div>', unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Performance Analytics")
        
        if show_analytics and st.session_state.performance_data["timestamp"]:
            df = pd.DataFrame(st.session_state.performance_data)
            df['time'] = pd.to_datetime(df['timestamp'], unit='s')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Gesture Distribution")
                fig1 = px.pie(df, names='gesture', title='Gesture Recognition Frequency')
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                st.markdown("### Confidence Levels")
                fig2 = px.box(df, x='gesture', y='confidence', title='Confidence Distribution by Gesture')
                st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("### Processing Time Over Time")
            fig3 = px.line(df, x='time', y='processing_time', title='Processing Time Trend')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No analytics data available yet. Use the real-time recognition to gather data.")
    
    # Add footer
    show_author_info()

if __name__ == "__main__":
    main()
