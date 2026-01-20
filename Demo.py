import cv2
import joblib
import numpy as np
import mediapipe as mp
import warnings

# --- SILENCE WARNINGS ---
# This stops the "UserWarning: X does not have valid feature names" spam
warnings.filterwarnings("ignore", category=UserWarning)

# --- STANDARD IMPORT ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION ---
MODEL_PATH = 'gesture_model.pkl'

def normalize_landmarks(landmarks):
    """
    Must match the EXACT logic used during training.
    """
    wrist = landmarks.landmark[0]
    wrist_x, wrist_y = wrist.x, wrist.y

    coords = []
    for lm in landmarks.landmark:
        relative_x = lm.x - wrist_x
        relative_y = lm.y - wrist_y
        coords.extend([relative_x, relative_y])
        
    return coords

def main():
    # 1. Load the trained brain
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("ERROR: Model file not found. Run TrainModel.py first!")
        exit()
    print("Model loaded successfully!")

    # 2. Setup Webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # 3. Setup MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False, # Faster for video
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Starting Webcam... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Flip frame for mirror effect and convert to RGB
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(image_rgb)

        # If a hand is found...
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # A. Draw the skeleton
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )

                # B. Extract features (Math)
                features = normalize_landmarks(hand_landmarks)
                
                # C. Make Prediction
                # reshape(1, -1) tells the model "this is just one sample"
                prediction = model.predict([features])[0]
                probabilities = model.predict_proba([features])[0]
                confidence = max(probabilities)

                # D. Display Result
                # If confidence is high, show Green. Low? Red.
                color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
                
                # Format text: "Like (99.5%)"
                text = f"{prediction} ({confidence*100:.1f}%)"
                
                # Draw text above the wrist
                wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])
                
                # Outline for better readability
                cv2.putText(frame, text, (wrist_x - 50, wrist_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(frame, text, (wrist_x - 50, wrist_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show the window
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()