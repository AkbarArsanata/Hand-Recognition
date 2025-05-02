import json
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from datetime import datetime

# Load konfigurasi model
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Load model dan parameter
model = load_model(config['model_path'])
class_names = config['class_names']
img_size = tuple(config['img_size'])

# Fungsi preprocessing
def preprocess_image(img):
    img = cv2.resize(img, img_size)
    if config['preprocessing']['normalization'] == 'divide_by_255':
        img = img / 255.0
    return img

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
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
            gesture = class_names[predicted_class]
            confidence = np.max(prediction)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{gesture} ({confidence:.2f})", 
                       (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 255, 0), 2)
    
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()