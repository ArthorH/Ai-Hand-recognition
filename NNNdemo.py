import cv2
import joblib
import numpy as np
import mediapipe as mp
import warnings

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODEL_PATH = 'gesture_model.pkl'

# --- SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def normalize_landmarks(landmarks):
    wrist = landmarks.landmark[0]
    wrist_x, wrist_y = wrist.x, wrist.y
    coords = []
    for lm in landmarks.landmark:
        # Relative coords
        coords.extend([lm.x - wrist_x, lm.y - wrist_y])
    return coords

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def draw_layer(img, activations, x_center, y_start, height, layer_name, cell_size=8, grid_width=1):
    """
    Draws a column or grid of neurons. 
    Brightness = Activation intensity.
    """
    n_neurons = len(activations)
    
    # Normalize activations for visualization (0 to 1) relative to this layer
    if np.max(activations) > 0:
        norm_acts = activations / np.max(activations)
    else:
        norm_acts = activations

    # Calculate layout
    cols = grid_width
    rows = int(np.ceil(n_neurons / cols))
    
    y_step = height // rows
    x_step = 15 # Spacing between grid columns
    
    for i in range(n_neurons):
        row = i // cols
        col = i % cols
        
        # Position
        cx = x_center + (col * x_step) - ((cols*x_step)//2)
        cy = y_start + (row * y_step) + (y_step//2)
        
        # Color: Dark Blue (inactive) -> Bright Cyan/White (active)
        intensity = int(norm_acts[i] * 255)
        color = (intensity, intensity, 255) # BGR
        
        # Draw Neuron
        cv2.circle(img, (cx, cy), cell_size, color, -1)
        # Draw faint outline
        cv2.circle(img, (cx, cy), cell_size, (50, 50, 50), 1)

    # Label
    cv2.putText(img, layer_name, (x_center - 30, y_start - 10), 
                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)

def main():
    print(f"Loading Neural Network from {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
        
        # Check if it's actually a Neural Network (MLP)
        if not hasattr(model, 'coefs_'):
            print("ERROR: This model is not a Neural Network (MLP).")
            print("Please run the 'TrainNeuralNet.py' script provided earlier.")
            exit()
            
        weights = model.coefs_
        biases = model.intercepts_
        classes = model.classes_
        print("Brain Loaded Successfully!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        exit()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Create Visualization Panel (Right side)
        panel_w = 500
        vis_panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Default empty activations
        layer_activations = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton on main frame
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 1. Input Layer (42 numbers)
                input_vector = np.array(normalize_landmarks(hand_landmarks))
                layer_activations.append(input_vector)

                # 2. Forward Pass (Manually calculating the layers)
                # Layer 1
                hidden1 = relu(np.dot(input_vector, weights[0]) + biases[0])
                layer_activations.append(hidden1)
                
                # Layer 2
                hidden2 = relu(np.dot(hidden1, weights[1]) + biases[1])
                layer_activations.append(hidden2)
                
                # Output Layer
                final_logits = np.dot(hidden2, weights[2]) + biases[2]
                output_probs = softmax(final_logits)
                layer_activations.append(output_probs)
                
                # Get winner
                winner_idx = np.argmax(output_probs)
                winner_label = classes[winner_idx]
                confidence = output_probs[winner_idx]
                
                # Display Prediction on Camera
                cv2.putText(frame, f"{winner_label} ({confidence*100:.0f}%)", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # --- DRAW THE VISUALIZATION ---
        if layer_activations:
            # Layout X-coordinates
            margin = 50
            layer_centers = [margin, 150, 250, 400]
            
            # Input Layer (42 nodes)
            draw_layer(vis_panel, np.abs(layer_activations[0]), layer_centers[0], 50, h-100, "Input (42)", 3, grid_width=2)
            
            # Hidden Layer 1 (100 nodes)
            draw_layer(vis_panel, layer_activations[1], layer_centers[1], 50, h-100, "Hidden 1 (100)", 3, grid_width=5)
            
            # Hidden Layer 2 (50 nodes)
            draw_layer(vis_panel, layer_activations[2], layer_centers[2], 50, h-100, "Hidden 2 (50)", 4, grid_width=3)
            
            # Output Layer (Classes)
            # We draw this as a bar chart instead of dots for readability
            y_start = 80
            for i, prob in enumerate(layer_activations[3]):
                label = classes[i]
                y_pos = y_start + (i * 30)
                
                # Bar color
                color = (0, 255, 0) if prob > 0.5 else (50, 50, 50)
                
                # Text
                cv2.putText(vis_panel, label, (layer_centers[3], y_pos), 
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (200,200,200), 1)
                
                # Bar
                bar_len = int(prob * 100)
                cv2.rectangle(vis_panel, (layer_centers[3] + 80, y_pos-10), 
                              (layer_centers[3] + 80 + bar_len, y_pos+5), color, -1)

        # Stitch them together
        final_view = np.hstack((frame, vis_panel))
        
        cv2.imshow('Live Neural Network Activations', final_view)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()