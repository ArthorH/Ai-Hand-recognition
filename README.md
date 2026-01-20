HaGrid Gesture Recognition & Neural Network Visualizer

<img width="1452" height="610" alt="image" src="https://github.com/user-attachments/assets/d9fa3b47-ccc6-49ba-a290-28917ed095db" />




A lightweight "evening project" exploring hand gesture recognition using the HaGrid 500k dataset. It extracts hand landmarks via MediaPipe and feeds them into a custom Neural Network, visualizing the neuron activations in real-time.

⚡ Fast Facts

    Dataset: Subsampled from HaGrid 500k (Hand Gesture Recognition).

    Tech Stack: OpenCV, MediaPipe, Scikit-Learn.

    Model: Multi-Layer Perceptron (MLP) with 2 hidden layers (100 & 50 neurons).

    The Cool Part: NNNdemo.py manually calculates the forward pass to visualize the "brain's" hidden layers live.

📂 File Guide

File	Function
NNNdemo.py	Start here. Runs the webcam demo with the live Neural Net visualizer.
TrainNeuralNetwork.py	Trains the MLP (Neural Net) used for the visualization.
DataSetGenerator.py	Parses HaGrid JSONs + Images → Normalized CSV landmarks.
TrainModel.py	Alternative training script using Random Forest (higher accuracy, no viz).
Demo.py	Minimal webcam inference script (no visualizer).

🚦 Quick Start

    Install:
    Bash

    pip install opencv-python mediapipe scikit-learn pandas joblib

    Train (or use pre-trained):

        You need the HaGrid dataset to generate fresh data.

        Run DataSetGenerator.py → TrainNeuralNetwork.py.

    Run:
    Bash

    python NNNdemo.py

🧠 How It Works

    Input: MediaPipe extracts 21 hand landmarks (x,y).

    Normalize: Coordinates are relative to the wrist to ensure translation invariance.

    Process: An MLP (42 inputs → 100 hidden → 50 hidden → Classes) classifies the gesture.

    Visualize: The demo app draws the raw activation intensity of every neuron during the forward pass.
