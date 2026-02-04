Here is an updated, more professional version of your README.md. I’ve refined the structure to make it look like a high-quality GitHub repository, emphasizing the technical "Cool Part" and the data pipeline.
🖐️ HaGrid Gesture Recognition & Neural Net Visualizer

<img width="1003" height="428" alt="image" src="https://github.com/user-attachments/assets/23ed9cf8-4214-4bce-97ca-dc3b50c93077" />
<img width="1003" height="276" alt="image" src="https://github.com/user-attachments/assets/effdb722-f7ba-43cf-bd41-819e111e848f" />


A lightweight "evening project" designed to pull back the curtain on the "Black Box" of AI. This project uses a subsampled HaGrid 500k dataset to train a Multi-Layer Perceptron (MLP) and visualizes the neuron activations in real-time as you gesture.
⚡ Fast Facts

    Dataset: Subsampled from HaGrid 500k (Hand Gesture Recognition Image Dataset).

    Feature Extraction: MediaPipe (converts raw images into 21 hand landmarks).

    Tech Stack: OpenCV, MediaPipe, Scikit-Learn, Pandas.

    Model: MLP with 2 hidden layers (128 & 64 neurons).

    The "Brain" Visualization: NNNdemo.py manually calculates the forward pass (Matrix Multiplication + ReLU) to render the intensity of hidden layers live on screen.

📂 File Guide
File	Function
NNNdemo.py	Primary Demo. Runs webcam with live Neural Net neuron visualization.
TrainNeuralNetwork.py	Trains the MLP model. Optimizes weights for visualization.
DataSetGenerator.py	The Pipeline: Parses HaGrid JSONs + Images → Normalized CSV landmarks.
TrainModel.py	Alternative: Trains a Decision Tree/Random Forest (higher accuracy, limited viz).
Demo.py	Minimal webcam inference script (standard output, no visualizer).
🧠 How It Works

    Extraction: MediaPipe identifies 21 points (x, y) on the hand.

    Normalization: All coordinates are calculated relative to the Wrist (Landmark 0). This makes the model "translation invariant" (it doesn't matter where your hand is in the frame).

    Inference: The MLP (42 inputs → 128 hidden → 64 hidden → N Classes) predicts the gesture.

    Live Visualization: Every neuron is drawn as a circle. Its brightness corresponds to its activation value after the ReLU function. When you change gestures, you can see different "paths" in the brain light up.

🚦 Quick Start
1. Requirements
Bash

pip install opencv-python mediapipe scikit-learn pandas joblib

2. Prepare Data & Train

Note: You need the HaGrid dataset sample locally to generate fresh data.

    Run DataSetGenerator.py to process images into train_gestures.csv.

    Run TrainNeuralNetwork.py to generate mlp_model.pkl.

3. Run Visualization
Bash

python NNNdemo.py

🛠️ Model Parameters (MLP)

    Hidden Layers: (128, 64)

    Activation: ReLU

    Optimizer: Adam

    Output: Softmax (Probability distribution over gestures)
