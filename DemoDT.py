import cv2
import joblib
import numpy as np
import mediapipe as mp
import warnings
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier

# --- KONFIGURACJA ---
MODEL_PATH = 'dt_model.pkl' 
warnings.filterwarnings("ignore")

# --- NAZWY CECH ---
LANDMARK_NAMES = [
    "Wrist", "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip"
]

FEATURE_NAMES = []
for name in LANDMARK_NAMES:
    FEATURE_NAMES.append(f"{name}.X")
    FEATURE_NAMES.append(f"{name}.Y")

# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    wrist = landmarks.landmark[0]
    wrist_x, wrist_y = wrist.x, wrist.y
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x - wrist_x, lm.y - wrist_y])
    return np.array(coords, dtype=np.float32)

def draw_node(img, node_id, tree, current_path, x, y, dx, dy, depth, max_depth, classes):
    if depth > max_depth:
        return

    # Czy ten węzeł jest na aktywnej ścieżce?
    is_active = node_id in current_path
    
    # Kolory (Aktywny = Zielony, Nieaktywny = Ciemny szary)
    bg_color = (0, 200, 0) if is_active else (40, 40, 40)
    border_color = (255, 255, 255) if is_active else (100, 100, 100)
    text_color = (255, 255, 255) if is_active else (150, 150, 150)
    
    feature_idx = tree.feature[node_id]
    threshold = tree.threshold[node_id]
    
    is_leaf = (feature_idx == _tree.TREE_UNDEFINED)
    
    # Rozmiary ramek
    box_w = 120 if is_leaf else 40
    box_h = 25
    
    # Pozycje
    x = int(x)
    y = int(y)
    top_left = (x - box_w//2, y - box_h//2)
    bottom_right = (x + box_w//2, y + box_h//2)
    
    # Rysowanie pudełka
    cv2.rectangle(img, top_left, bottom_right, bg_color, -1)
    cv2.rectangle(img, top_left, bottom_right, border_color, 1)
    
    if not is_leaf:
        # --- WĘZEŁ DECYZYJNY ---
        # Skracanie nazw, żeby się mieściły
        if feature_idx < len(FEATURE_NAMES):
            fname = FEATURE_NAMES[feature_idx]
            fname = fname.replace("Thumb", "Tmb").replace("Index", "Idx")
            fname = fname.replace("Middle", "Mid").replace("Ring", "Rng").replace("Pinky", "Pnk")
        else:
            fname = f"F{feature_idx}"

        # Tekst nad pudełkiem
        cv2.putText(img, fname, (x - 40, y - 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        # Warunek pod pudełkiem
        cv2.putText(img, f"<= {threshold:.2f}", (x - 40, y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Linie do dzieci
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        
        new_dx = dx / 2
        child_y = y + dy
        left_x = x - dx
        right_x = x + dx
        
        # Kolor linii
        l_col = (0, 255, 0) if (left_child in current_path) else (60, 60, 60)
        r_col = (0, 255, 0) if (right_child in current_path) else (60, 60, 60)
        
        cv2.line(img, (x, y + box_h//2), (int(left_x), int(child_y - box_h//2)), l_col, 1)
        cv2.line(img, (x, y + box_h//2), (int(right_x), int(child_y - box_h//2)), r_col, 1)
        
        # Rekurencja
        draw_node(img, left_child, tree, current_path, left_x, child_y, new_dx, dy, depth+1, max_depth, classes)
        draw_node(img, right_child, tree, current_path, right_x, child_y, new_dx, dy, depth+1, max_depth, classes)

    else:
        # --- LIŚĆ (WYNIK) ---
        if tree.n_outputs == 1:
            value = tree.value[node_id][0]
        else:
            value = tree.value[node_id]

        class_idx = np.argmax(value)
        try:
            class_name = str(classes[class_idx])
        except:
            class_name = f"Class {class_idx}"
        
        # Wyśrodkowanie tekstu
        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx = x - text_size[0] // 2
        ty = y + text_size[1] // 2
        
        cv2.putText(img, class_name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def get_decision_path(tree, X_sample):
    path = []
    node_id = 0
    path.append(node_id)
    
    while tree.feature[node_id] != _tree.TREE_UNDEFINED:
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        
        if X_sample[feature] <= threshold:
            node_id = tree.children_left[node_id]
        else:
            node_id = tree.children_right[node_id]
        path.append(node_id)
    return path

def main():
    print("--- STARTING APP ---")
    print(f"Loading model: {MODEL_PATH}")
    
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {MODEL_PATH}!")
        return
    except Exception as e:
        print(f"BŁĄD przy ładowaniu modelu: {e}")
        return

    # --- DETEKCJA TYPU MODELU ---
    viz_tree = None
    if hasattr(model, 'tree_'):
        print("Model wykryty jako: Single Decision Tree")
        viz_tree = model.tree_
    elif hasattr(model, 'estimators_'):
        print("Model wykryty jako: Random Forest (Wizualizuje pierwsze drzewo)")
        viz_tree = model.estimators_[0].tree_
    else:
        print("BŁĄD: Model nie jest drzewem ani lasem losowym. Nie można wizualizować struktury.")
        # Nie wychodzimy, żeby chociaż pokazać predykcję, ale wizualizacji nie będzie
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("BŁĄD: Nie można otworzyć kamery.")
        return

    print("Kamera uruchomiona. Naciśnij 'q' aby wyjść.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Panel wizualizacji (szeroki)
        vis_w = 1200
        vis_panel = np.zeros((h, vis_w, 3), dtype=np.uint8)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        prediction_text = "Czekam na dlon..."
        color = (0, 0, 255) # Czerwony domyślnie

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                features = normalize_landmarks(hand_landmarks)
                
                # 1. Prawdziwa predykcja (działa dla Tree i Forest)
                try:
                    pred = model.predict([features])[0]
                    prediction_text = f"GEST: {pred}"
                    color = (0, 255, 0) # Zielony jak wykryje
                except Exception as e:
                    prediction_text = "Blad predykcji"
                    print(e)

                # 2. Wizualizacja ścieżki (jeśli mamy drzewo)
                if viz_tree is not None:
                    try:
                        path = get_decision_path(viz_tree, features)
                        draw_node(vis_panel, 0, viz_tree, path, 
                                  vis_w // 2, 50, vis_w // 4.5, 80, 
                                  0, 5, model.classes_)
                    except Exception as e:
                        print(f"Blad wizualizacji: {e}")

        # Rysowanie wyniku na kamerze (Duży napis na górze)
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, prediction_text, (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        # Łączenie obrazów
        combined = np.hstack((frame, vis_panel))
        
        # Skalowanie do ekranu (dopasuj fx, fy jeśli za duże/za małe)
        output = cv2.resize(combined, (0, 0), fx=0.7, fy=0.7)
        
        cv2.imshow('AI Logic Viewer', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()