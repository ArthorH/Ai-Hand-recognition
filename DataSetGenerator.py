import os
import json
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- STANDARD IMPORT ---
import mediapipe as mp

# --- CONFIGURATION ---
JSON_DIR = r'hagrid-sample-500k-384p\ann_train_val'
IMAGE_DIR = r'hagrid-sample-500k-384p\hagrid_500k'
OUTPUT_DIR = 'processed_data'
LIMIT_PER_CLASS = 1500 

# --- SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands  # This should work now!
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

def normalize_landmarks(landmarks):
    wrist = landmarks.landmark[0]
    wrist_x, wrist_y = wrist.x, wrist.y
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x - wrist_x, lm.y - wrist_y])
    return coords

def parse_annotations(json_dir):
    mapping = {}
    print(f"Reading JSON annotations from {json_dir}...")
    if not os.path.exists(json_dir):
        print(f"ERROR: JSON directory not found: {json_dir}")
        exit()

    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            label = filename.replace('.json', '') 
            path = os.path.join(json_dir, filename)
            with open(path, 'r') as f:
                data = json.load(f)
            for image_uuid in data.keys():
                mapping[image_uuid] = label
                
    print(f"Loaded annotations. Mapped {len(mapping)} images.")
    return mapping

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    id_to_label = parse_annotations(JSON_DIR)
    
    data_records = []
    class_counts = {} 
    
    print("Starting Feature Extraction...")
    print(f"Scanning directory: {IMAGE_DIR}")
    
    if not os.path.exists(IMAGE_DIR):
        print(f"ERROR: Image directory not found: {IMAGE_DIR}")
        exit()

    valid_extensions = ('.jpg', '.jpeg', '.png')
    processed_total = 0

    for root, dirs, files in os.walk(IMAGE_DIR):
        for img_file in files:
            if not img_file.lower().endswith(valid_extensions):
                continue

            uuid = os.path.splitext(img_file)[0]
            if uuid not in id_to_label:
                continue
                
            label = id_to_label[uuid]

            if LIMIT_PER_CLASS:
                current_count = class_counts.get(label, 0)
                if current_count >= LIMIT_PER_CLASS:
                    continue 
            
            img_path = os.path.join(root, img_file)
            image = cv2.imread(img_path)
            if image is None: continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row_features = normalize_landmarks(hand_landmarks)
                    row_features.insert(0, label) 
                    data_records.append(row_features)
                    
                    class_counts[label] = class_counts.get(label, 0) + 1
                    processed_total += 1
                    
                    if processed_total % 100 == 0:
                        print(f"Processed {processed_total} hands... (Last: {label})")

    print("\nExtraction complete!")
    print("Class distribution:", class_counts)
    
    cols = ['label']
    for i in range(21):
        cols.extend([f'x{i}', f'y{i}'])
        
    df = pd.DataFrame(data_records, columns=cols)
    df.dropna(inplace=True)
    
    if len(df) > 0:
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_gestures.csv'), index=False)
        test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_gestures.csv'), index=False)
        print(f"SUCCESS! Data saved to '{OUTPUT_DIR}'")
    else:
        print("ERROR: No landmarks were extracted.")