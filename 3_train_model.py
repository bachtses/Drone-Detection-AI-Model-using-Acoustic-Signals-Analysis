import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, BatchNormalization, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import random
from tqdm import tqdm

# Configure GPU memory growth if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# ---------------------- CONSTANTS ----------------------
DATASET_PATH = "dataset"
SAMPLE_RATE = 16000  # Standard sample rate
AUDIO_LENGTH = 2     # 2-second clips
N_MFCC = 40          # Number of MFCC features to extract
NUM_CLASSES = 2      # Binary classification


# ---------------------- DATA AUGMENTATION FUNCTIONS ----------------------
def add_gaussian_noise(audio, noise_level=0.005):
    return audio + noise_level * np.random.randn(len(audio))

def time_shift(audio, shift_max=0.2):
    shift = int(shift_max * SAMPLE_RATE * (np.random.rand() - 0.5))
    return np.roll(audio, shift)

def random_volume(audio, min_gain=0.8, max_gain=1.2):
    return audio * np.random.uniform(min_gain, max_gain)

def bandpass_filter(audio, lowcut=100, highcut=7900, sr=SAMPLE_RATE, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, audio)

# ---------------------- FEATURE EXTRACTION (MFCC) ----------------------
def extract_mfcc_features(audio, sr):

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # shape (40, n_frames)
    mfcc = mfcc.T   # shape (n_frames, 40)  
    return (mfcc - np.mean(mfcc)) / np.std(mfcc)  


# ---------------------- LOAD AUDIO DATA WITH AUGMENTATION ----------------------
def load_audio_data(folder_path, label, max_files=None, augment=True):
    data, labels = [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    print("\nTotal samples in Folder:", folder_path, len(files))

    if max_files:
        files = random.sample(files, min(len(files), max_files))

    print(f"Processing: {folder_path} (Label: {label})")

    for file in tqdm(files, desc=f"Processing {folder_path.split('/')[-1]}", unit="file"):
        file_path = os.path.join(folder_path, file)
        #audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        audio, sr = librosa.load(file_path, sr=None)  # Load at native sample rate
        if sr != SAMPLE_RATE:
            #print(f"[INFO] Resampling {file} from {sr} Hz to {SAMPLE_RATE} Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE



        target_length = SAMPLE_RATE * AUDIO_LENGTH
        audio = np.pad(audio, (0, max(0, target_length - len(audio))))[:target_length]

        # Extract MFCC Features from original sample
        mfcc_features = extract_mfcc_features(audio, sr)
        data.append(mfcc_features)
        labels.append(label)

        # ---------------------- DATA AUGMENTATION ----------------------
        if augment:
            AUGMENTATION_FACTOR = 2
            for _ in range(AUGMENTATION_FACTOR):
                augmented_audio = audio.copy()
                if random.random() < 0.3: 
                    augmented_audio = add_gaussian_noise(augmented_audio)
                if random.random() < 0.3: 
                    augmented_audio = time_shift(augmented_audio)
                if random.random() < 0.3: 
                    augmented_audio = random_volume(augmented_audio)
                if random.random() < 0.3: 
                    augmented_audio = bandpass_filter(augmented_audio, sr=sr)

                augmented_mfcc = extract_mfcc_features(augmented_audio, sr)
                data.append(augmented_mfcc)
                labels.append(label)
    return np.array(data), np.array(labels)


# ---------------------- LOAD DATA ----------------------
print("Loading dataset...\n")
X_drone, y_drone = load_audio_data(os.path.join(DATASET_PATH, "train", "drone"), label=1, augment=True)
X_background, y_background = load_audio_data(os.path.join(DATASET_PATH, "train", "background"), label=0, augment=True)

X = np.concatenate([X_drone, X_background], axis=0)
y = np.concatenate([y_drone, y_background], axis=0)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]
# Reshape features to add a channel dimension (for Conv1D)
#X = X.reshape(X.shape[0], X.shape[1], 1)


print(f"\nTotal training set: {X.shape}\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------- MODEL ----------------------
print("\nBuilding model...\n")

# Suppose X has shape (num_samples, n_frames, n_mfcc)
print(X.shape)  # should be e.g. (4000, 94, 40)

# model:
model = Sequential([
    Conv1D(64, kernel_size=7, activation='relu',
           kernel_regularizer=l2(0.002),
           input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),

    Conv1D(128, kernel_size=3, activation='relu',
           kernel_regularizer=l2(0.002)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),

    Conv1D(256, kernel_size=3, activation='relu',
           kernel_regularizer=l2(0.002)),
    BatchNormalization(),
    GlobalAveragePooling1D(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test), 
                    callbacks=[reduce_lr, early_stopping])

model.save('model.h5')


# ---------------------- PLOTS ----------------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Background', 'Drone']))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Background', 'Drone'], yticklabels=['Background', 'Drone'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.show()
