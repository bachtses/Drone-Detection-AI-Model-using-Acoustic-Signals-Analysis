A real-time, AI-powered acoustic detection system for identifying drones using distributed microphone arrays, edge computing (Raspberry Pi 5), and long-range wireless communication (LoRa). The system is designed to operate in outdoor environments with real-world noise and varying drone behaviors.

**Features**
1D Convolutional Neural Network (CNN) trained on real-world audio samples for binary classification (drone / no drone)
ReSpeaker 4-Mic Arrays with Angle-of-Arrival (AoA) estimation
Distributed Edge Nodes using Raspberry Pi 5 for real-time inference
LoRa Communication for long-range, low-bandwidth wireless data transfer
Fusion Aggregator for combining predictions from multiple nodes
Central Node Dashboard with real-time spectrogram streaming and detection visualization (Flask-based)
JSON-based API integration with external platforms for live event reporting

**System Components**
Peripheral Nodes (Raspberry Pi + ReSpeaker + LoRa): Perform local inference and send detection metadata
Aggregator Node: Gathers predictions, applies confidence fusion logic, and forwards results to the central node
Central Node: Displays detections and streams spectrograms to UI / dashboard
AI Model: Trained with MFCC features, 3×Conv1D layers, >95% accuracy on validation

**Evaluation**
Real-world field tested with over 6500 labeled audio samples
Achieved F1 Score: 91.2%, Precision: 94.2%, Recall: 88.5%, and Accuracy: 90.2% during field deployment
Tested with DJI Mavic 3 drone under varying distances, angles, and environmental noise

**Repository Structure**
/peripheral_node_raspberry.py     # Real-time inference node script
/aggregator.py                    # Fusion node to combine predictions
/central_node.py                  # Flask server for visualization and control
/model_training/                  # Scripts for model training & preprocessing
/model.tflite                     # Final converted TFLite model for edge inference
/setup/                           # LoRa setup scripts and systemd autostart files

**Installation & Deployment**
Raspberry Pi OS + Python virtual environments
Installing dependencies (librosa, tflite-runtime, etc.)
LoRa module configuration and AT command automation
Autostart on boot with systemd services
