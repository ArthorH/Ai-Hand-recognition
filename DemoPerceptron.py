import cv2
import joblib
import numpy as np
import mediapipe as mp
import warnings

# --- KONFIGURACJA ---
MODEL_PATH = 'mlp_model.pkl'  # Upewnij się, że to nazwa Twojego modelu
warnings.filterwarnings("ignore")

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- FUNKCJE POMOCNICZE ---

def normalize_landmarks(landmarks):
    """Konwertuje punkty dłoni na współrzędne względne (względem nadgarstka)"""
    wrist = landmarks.landmark[0]
    wrist_x, wrist_y = wrist.x, wrist.y
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x - wrist_x, lm.y - wrist_y])
    return coords

def relu(x):
    """Funkcja aktywacji ReLU (zeruje wartości ujemne)"""
    return np.maximum(0, x)

def softmax(x):
    """Funkcja Softmax (zamienia wyniki na prawdopodobieństwa)"""
    e_x = np.exp(x - np.max(x)) # Odejmujemy max dla stabilności numerycznej
    return e_x / e_x.sum()

def draw_network_grid(img, activations, x_start, y_start, width, height, layer_name):
    """
    Rysuje warstwę neuronów w formie siatki (grid).
    Jaśniejszy kolor = silniejsza aktywacja.
    """
    n_neurons = len(activations)
    
    # Normalizacja do wyświetlania (0.0 - 1.0)
    max_val = np.max(activations)
    if max_val > 0:
        norm_acts = activations / max_val
    else:
        norm_acts = activations

    # Obliczanie układu siatki (np. dla 128 neuronów zrób kwadrat)
    cols = int(np.sqrt(n_neurons)) + 1
    rows = (n_neurons // cols) + 1
    
    cell_w = width // cols
    cell_h = height // rows
    radius = min(cell_w, cell_h) // 3
    
    cv2.putText(img, f"{layer_name} ({n_neurons})", (x_start, y_start - 10), 
                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)

    for i in range(n_neurons):
        r = i // cols
        c = i % cols
        
        cx = x_start + (c * cell_w) + (cell_w // 2)
        cy = y_start + (r * cell_h) + (cell_h // 2)
        
        # Kolor: Od ciemnego granatu (0) do jasnego cyjanu (1)
        intensity = int(norm_acts[i] * 255)
        # BGR Format
        color = (intensity, intensity, 50 + (intensity // 2)) 
        
        # Rysuj neuron
        cv2.circle(img, (cx, cy), radius, color, -1)
        # Obwódka
        cv2.circle(img, (cx, cy), radius, (30, 30, 30), 1)

def draw_output_bars(img, probs, classes, x_start, y_start, width):
    """Rysuje wyniki końcowe jako paski"""
    cv2.putText(img, "OUTPUT (Classes)", (x_start, y_start - 15), 
                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
    
    bar_height = 25
    gap = 10
    
    # Znajdź zwycięzcę
    winner_idx = np.argmax(probs)
    
    for i, (prob, label) in enumerate(zip(probs, classes)):
        y = y_start + i * (bar_height + gap)
        
        # Kolor paska: Zielony dla zwycięzcy, szary dla reszty
        color = (0, 255, 0) if i == winner_idx and prob > 0.5 else (100, 100, 100)
        
        # Nazwa klasy
        cv2.putText(img, label, (x_start, y + 20), 
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 1)
        
        # Pasek
        bar_max_width = width - 120
        cur_width = int(bar_max_width * prob)
        cv2.rectangle(img, (x_start + 110, y + 5), (x_start + 110 + cur_width, y + 20), color, -1)
        
        # Procenty
        cv2.putText(img, f"{prob*100:.0f}%", (x_start + 110 + cur_width + 5, y + 20), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)

# --- GLÓWNA PĘTLA ---
def main():
    print(f"🔄 Ładowanie modelu z {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
        
        # Sprawdzamy czy to MLP (Sieć Neuronowa)
        if not hasattr(model, 'coefs_'):
            print("❌ BŁĄD: To nie jest model sieci neuronowej (MLP).")
            print("To prawdopodobnie Drzewo lub Las. Uruchom skrypt trenujący MLP.")
            return

        # Wyciągamy wagi i biasy (to jest mózg sieci)
        weights = model.coefs_
        biases = model.intercepts_
        classes = model.classes_
        print("✅ Model załadowany! Naciśnij 'q' aby wyjść.")
        
    except FileNotFoundError:
        print(f"❌ BŁĄD: Nie znaleziono pliku {MODEL_PATH}.")
        return

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Odbicie lustrzane dla naturalnego odczucia
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Tworzymy panel wizualizacji (czarne tło po prawej)
        vis_width = 600
        vis_panel = np.zeros((h, vis_width, 3), dtype=np.uint8)
        
        # MediaPipe Processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        activations = [] # Tutaj będziemy zbierać stany neuronów

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 1. WARSTWA WEJŚCIOWA (Input)
                # Pobieramy dane tak jak przy treningu
                input_data = np.array(normalize_landmarks(hand_landmarks))
                activations.append(input_data)

                # 2. PROPAGACJA W PRZÓD (Manual Forward Pass)
                # Symulujemy działanie sieci krok po kroku, żeby widzieć środek
                
                # Warstwa ukryta 1
                # Wzór: A = ReLU(Input * Wagi + Bias)
                hidden1 = relu(np.dot(input_data, weights[0]) + biases[0])
                activations.append(hidden1)
                
                # Warstwa ukryta 2 (jeśli istnieje)
                current_input = hidden1
                for i in range(1, len(weights) - 1):
                    hidden_next = relu(np.dot(current_input, weights[i]) + biases[i])
                    activations.append(hidden_next)
                    current_input = hidden_next
                
                # Warstwa Wyjściowa (Output)
                # Wzór: O = Softmax(Hidden * Wagi + Bias)
                final_logits = np.dot(current_input, weights[-1]) + biases[-1]
                output_probs = softmax(final_logits)
                activations.append(output_probs)

        # --- RYSOWANIE WIZUALIZACJI ---
        if activations:
            # Rozkład elementów na panelu
            # Input Layer (top)
            draw_network_grid(vis_panel, np.abs(activations[0]), 20, 50, 150, 150, "Input (42)")
            
            # Hidden Layers (middle)
            # Rysujemy max 2 warstwy ukryte dla czytelności
            hidden_x = 200
            if len(activations) > 2:
                 draw_network_grid(vis_panel, activations[1], hidden_x, 50, 180, 250, "Hidden 1")
                 if len(activations) > 3:
                     draw_network_grid(vis_panel, activations[2], hidden_x + 200, 50, 180, 250, "Hidden 2")
            
            # Output Layer (bottom) - jako paski
            draw_output_bars(vis_panel, activations[-1], classes, 20, 350, vis_width - 40)
        else:
            cv2.putText(vis_panel, "Pokaz dlon...", (50, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        # Łączenie obrazów (Kamera + Wizualizacja)
        combined = np.hstack((frame, vis_panel))
        
        cv2.imshow('Neural Network Live Internals', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()