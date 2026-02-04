import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- KONFIGURACJA ---
TRAIN_DATA_PATH = 'processed_data/train_gestures.csv'
TEST_DATA_PATH = 'processed_data/test_gestures.csv'
MLP_MODEL_PATH = 'mlp_model.pkl'
DT_MODEL_PATH = 'dt_model.pkl'

def plot_confusion_matrix(y_true, y_pred, model_name, class_names):
    """
    Rysuje znormalizowaną macierz pomyłek dla danego modelu.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names, normalize='true')

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='YlGnBu', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Znormalizowana Macierz Pomyłek - {model_name}\n(Wartości 0-1, gdzie 1 = 100% skuteczności)', fontsize=15)
    plt.xlabel('Predykcja (Co "widzi" model)')
    plt.ylabel('Rzeczywistość (Jaki gest był naprawdę)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    print("--- 1. Ładowanie Danych ---")
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        print("BŁĄD: Nie znaleziono plików CSV. Uruchom najpierw generator danych.")
        return

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']
    
    # Pobieramy listę nazw klas posortowaną alfabetycznie
    class_names = sorted(y_test.unique())

    print(f"Dane wczytane. Trening: {len(X_train)}, Test: {len(X_test)}")
    print(f"Liczba klas: {len(class_names)}\n")

    # --- 2. TRENING MODELI ---
    
    # A. Drzewo Decyzyjne (Decision Tree)
    print("⏳ Trenowanie Drzewa Decyzyjnego (DT)...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=12) # max_depth zapobiega overfittingowi
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    joblib.dump(dt_model, DT_MODEL_PATH)

    # B. Sieć Neuronowa (MLP)
    print("⏳ Trenowanie Sieci Neuronowej (MLP)... (To może chwilę potrwać)")
    # Używamy nieco większej sieci (128, 64) dla lepszej dokładności
    mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=42)
    mlp_model.fit(X_train, y_train)
    y_pred_mlp = mlp_model.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    joblib.dump(mlp_model, MLP_MODEL_PATH)

    # --- 3. PORÓWNANIE WYNIKÓW ---
    print("\n" + "="*40)
    print("       WYNIKI PORÓWNANIA")
    print("="*40)
    
    diff = acc_mlp - acc_dt
    
    print(f"Drzewo Decyzyjne (DT): {acc_dt:.2%}")
    print(f"Sieć Neuronowa (MLP):  {acc_mlp:.2%}")
    print("-" * 40)
    
    if diff > 0:
        print(f"🏆 WINNER: Sieć Neuronowa wygrywa o {diff:.2%}")
    elif diff < 0:
        print(f"🏆 WINNER: Drzewo Decyzyjne wygrywa o {abs(diff):.2%}")
    else:
        print("🤝 REMIS: Oba modele mają identyczną skuteczność.")

    # --- 4. GENEROWANIE MACIERZY POMYŁEK ---
    print("\nGenerowanie wykresów macierzy pomyłek...")
    
    # Wykres dla Drzewa
    plot_confusion_matrix(y_test, y_pred_dt, "Decision Tree", class_names)
    
    # Wykres dla MLP
    plot_confusion_matrix(y_test, y_pred_mlp, "MLP (Neural Network)", class_names)

    print("\nGotowe! Modele zapisano jako .pkl")

if __name__ == "__main__":
    main()