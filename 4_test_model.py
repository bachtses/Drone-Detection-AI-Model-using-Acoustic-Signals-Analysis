import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from tqdm import tqdm

# ---------------------- CONSTANTS ----------------------
DATASET_PATH = "dataset"
SAMPLE_RATE = 16000  # Standard sample rate
AUDIO_LENGTH = 2     # 2-second clips
N_MFCC = 40          # Number of MFCC features to extract

# ---------------------- FEATURE EXTRACTION (MFCC) ----------------------
def extract_mfcc_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # shape (40, n_frames)
    mfcc = mfcc.T   # shape (n_frames, 40)  
    return (mfcc - np.mean(mfcc)) / np.std(mfcc) #Normalize

# ---------------------- LOAD AUDIO DATA (WITHOUT AUGMENTATION) ----------------------
def load_audio_data(folder_path, label, max_files=None, augment=False):
    data, labels = [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    print("\nTotal samples in Folder:", folder_path, len(files))

    if max_files:
        files = files[:min(len(files), max_files)]

    print(f"Processing: {folder_path} (Label: {label})")
    for file in tqdm(files, desc=f"Processing {os.path.basename(folder_path)}", unit="file"):
        file_path = os.path.join(folder_path, file)
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        target_length = SAMPLE_RATE * AUDIO_LENGTH
        audio = np.pad(audio, (0, max(0, target_length - len(audio))))[:target_length]
        # Extract MFCC features
        mfcc_features = extract_mfcc_features(audio, sr)
        data.append(mfcc_features)
        labels.append(label)

    return np.array(data), np.array(labels)

# ---------------------- LOAD TEST DATA ----------------------
print("Loading test dataset...\n")
X_test_drone, y_test_drone = load_audio_data(os.path.join(DATASET_PATH, "test", "drone"), label=1, augment=False)
X_test_background, y_test_background = load_audio_data(os.path.join(DATASET_PATH, "test", "background"), label=0, augment=False)

# Combine drone and background samples
X_test = np.concatenate([X_test_drone, X_test_background], axis=0)
y_test = np.concatenate([y_test_drone, y_test_background], axis=0)

# Reshape the data to add a channel dimension (for Conv1D input)
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(f"\nTotal test set: {X_test.shape}\n")

# ---------------------- LOAD SAVED MODEL ----------------------
model = load_model('model.h5')
print("Model loaded successfully.")

# ---------------------- MAKE PREDICTIONS ----------------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\nTest Data Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Background', 'Drone']))

# ---------------------- PLOT CONFUSION MATRIX ----------------------
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Background', 'Drone'],
            yticklabels=['Background', 'Drone'])
plt.title('Confusion Matrix on Test Data')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
