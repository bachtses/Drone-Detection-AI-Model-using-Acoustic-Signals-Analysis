import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import tensorflow as tf

# ---------------------- CONSTANTS ----------------------
DATASET_PATH = "dataset"
SAMPLE_RATE = 16000
AUDIO_LENGTH = 2
N_MFCC = 40
TFLITE_MODEL_PATH = "model.tflite"

# ---------------------- FEATURE EXTRACTION ----------------------
def extract_mfcc_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    return (mfcc - np.mean(mfcc)) / np.std(mfcc)

# ---------------------- LOAD AUDIO DATA ----------------------
def load_audio_data(folder_path, label, max_files=None):
    data, labels = [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    print(f"\nTotal samples in Folder: {folder_path}: {len(files)}")

    if max_files:
        files = files[:min(len(files), max_files)]

    print(f"Processing: {folder_path} (Label: {label})")
    for file in tqdm(files, desc=f"Processing {os.path.basename(folder_path)}", unit="file"):
        file_path = os.path.join(folder_path, file)
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        target_length = SAMPLE_RATE * AUDIO_LENGTH
        audio = np.pad(audio, (0, max(0, target_length - len(audio))))[:target_length]
        mfcc_features = extract_mfcc_features(audio, sr)
        data.append(mfcc_features)
        labels.append(label)

    return np.array(data), np.array(labels)

# ---------------------- LOAD TEST DATA ----------------------
print("Loading test dataset...\n")
X_test_drone, y_test_drone = load_audio_data(os.path.join(DATASET_PATH, "test", "drone"), label=1)
X_test_background, y_test_background = load_audio_data(os.path.join(DATASET_PATH, "test", "background"), label=0)

X_test = np.concatenate([X_test_drone, X_test_background], axis=0)
y_test = np.concatenate([y_test_drone, y_test_background], axis=0)

print(f"\nTotal test set: {X_test.shape}\n")

# ---------------------- LOAD TFLITE MODEL ----------------------
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"TFLite model input shape: {input_details[0]['shape']}")

# ---------------------- MAKE PREDICTIONS ----------------------
y_pred = []

for i in tqdm(range(X_test.shape[0]), desc="Evaluating TFLite Model"):
    input_data = X_test[i].astype(np.float32)[np.newaxis, :, :]
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0][0]
    y_pred.append(int(output_data > 0.5))

y_pred = np.array(y_pred)

# ---------------------- METRICS & VISUALIZATION ----------------------
print("\nTest Data Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Background', 'Drone']))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Background', 'Drone'],
            yticklabels=['Background', 'Drone'])
plt.title('Confusion Matrix (TFLite Model)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()
