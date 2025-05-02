import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import pandas as pd

# Set page config with author information
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="👋",
    layout="wide",
    menu_items={
        'About': """
        ## Hand Gesture Recognition App
        **Author**: Ibrahim Akbar Arsanata  
        **LinkedIn**: [linkedin.com/in/ibrahim-akbar-arsanata](https://www.linkedin.com/in/ibrahim-akbar-arsanata)  
        **Email**: arsanataibrahim9@gmail.com
        """
    }
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
    .feature-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe Hands
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5)
    mediapipe_loaded = True
except ImportError:
    mediapipe_loaded = False
    st.warning("MediaPipe not available - running in demo mode")

# Gesture guide with emojis
GESTURE_GUIDE = {
    "palm": {"emoji": "✋", "desc": "Open hand with fingers extended"},
    "fist": {"emoji": "✊", "desc": "Closed fist"},
    "point": {"emoji": "👉", "desc": "Index finger pointing"},
    "ok": {"emoji": "👌", "desc": "Thumb and index finger making a circle"}
}

# Initialize performance tracking
if 'performance' not in st.session_state:
    st.session_state.performance = {
        "gestures": [],
        "timestamps": [],
        "confidences": []
    }

def process_frame(frame):
    if not mediapipe_loaded:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Simplified processing for demo
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            
            # Demo gesture classification
            gesture = "palm" if y_min < h/2 else "fist"
            confidence = 0.85
            
            # Store performance
            st.session_state.performance["gestures"].append(gesture)
            st.session_state.performance["timestamps"].append(time.time())
            st.session_state.performance["confidences"].append(confidence)
            
            # Draw results
            cv2.putText(frame, f"{GESTURE_GUIDE[gesture]['emoji']} {gesture}", 
                       (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def show_gesture_guide():
    st.header("Gesture Guide")
    cols = st.columns(2)
    for i, (gesture, info) in enumerate(GESTURE_GUIDE.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{info['emoji']} {gesture.title()}</h3>
                <p>{info['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    st.markdown('<p class="header">Hand Gesture Recognition</p>', unsafe_allow_html=True)
    
    # Sidebar with author info
    with st.sidebar:
        st.markdown("""
        ### Developer Info
        **Ibrahim Akbar Arsanata**  
        [LinkedIn](https://www.linkedin.com/in/ibrahim-akbar-arsanata)  
        arsanataibrahim9@gmail.com
        """)
        st.markdown("---")
        st.write("Settings will appear here when MediaPipe is available")
    
    # Main app tabs
    tab1, tab2 = st.tabs(["Recognition", "Guide"])
    
    with tab1:
        st.subheader("Real-Time Recognition")
        img_file_buffer = st.camera_input("Show your hand to the camera")
        
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            frame = np.array(image)
            processed_frame = process_frame(frame)
            st.image(processed_frame, use_column_width=True)
            
            if st.session_state.performance["gestures"]:
                last_gesture = st.session_state.performance["gestures"][-1]
                st.success(f"Detected: {GESTURE_GUIDE[last_gesture]['emoji']} {last_gesture}")
    
    with tab2:
        show_gesture_guide()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Developed by <strong>Ibrahim Akbar Arsanata</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
