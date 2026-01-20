import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier # <--- This is the Neural Net
from sklearn.metrics import accuracy_score

# Load Data
train_df = pd.read_csv('processed_data/train_gestures.csv')
test_df = pd.read_csv('processed_data/test_gestures.csv')

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print("Training Neural Network (Multi-Layer Perceptron)...")

# MLP = Multi-Layer Perceptron (Classic Neural Network)
# hidden_layer_sizes=(100, 50) means:
# Layer 1: 100 Neurons
# Layer 2: 50 Neurons
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Neural Network Accuracy: {acc*100:.2f}%")

joblib.dump(model, 'gesture_model.pkl')
print("Saved new Neural Network model to 'gesture_model.pkl'")