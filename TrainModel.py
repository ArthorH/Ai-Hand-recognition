import pandas as pd
import joblib  # To save the trained model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
TRAIN_DATA_PATH = 'processed_data/train_gestures.csv'
TEST_DATA_PATH = 'processed_data/test_gestures.csv'
MODEL_FILENAME = 'gesture_model.pkl'

def train_model():
    print("Loading data...")
    # 1. Load the CSVs you just created
    try:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError:
        print("ERROR: Could not find CSV files. Did you run the Generator?")
        exit()

    print(f"Training with {len(train_df)} examples.")
    print(f"Testing with {len(test_df)} examples.")

    # 2. Separate Features (X) and Labels (y)
    # We drop 'label' to get just the coordinates
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    # 3. Initialize the Model
    # Random Forest is great for tabular data like coordinates
    print("Training the Random Forest model... (this might take a minute)")
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 4. Train!
    model.fit(X_train, y_train)

    # 5. Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nSUCCESS! Model Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save the "Brain"
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model saved to '{MODEL_FILENAME}'. Ready for the webcam app!")

if __name__ == "__main__":
    train_model()