import os
import numpy as np
import librosa
import tensorflow as tf
import pyaudio
import time
from datetime import datetime
from scipy.io.wavfile import write

# ---------------------- CONSTANTS ----------------------
SAMPLE_RATE = 24000     # Must match training sample rate
AUDIO_LENGTH = 2        # 2-second clips (44100 samples)
N_MFCC = 40             # Number of MFCC features to extract
CHUNK = 1024            # Frames per buffer


# ---------------------- LOAD THE MODEL ----------------------
try:
    model = tf.keras.models.load_model('model.h5')
    print("\033[96mModel loaded successfully.\033[0m")
except Exception as e:
    print("\033[91mFailed to load the model:\033[0m", e)



# ---------------------- INITIALIZE AUDIO STREAM WITH DEVICE CHECK ----------------------
p = pyaudio.PyAudio()
device_index = None
respeaker_keyword = "respeaker 4 mic array"  # Case-insensitive search
print("\033[96mChecking for ReSpeaker microphone device:\033[0m")
for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if dev_info['maxInputChannels'] > 0 and respeaker_keyword in dev_info['name'].lower():
        device_index = i
        print(f"\033[96mFound ReSpeaker device: index {i}, name: {dev_info['name']}\033[0m")
        break
if device_index is None:
    print("\033[91mError: ReSpeaker microphone not found. Please check that it is connected.\033[0m")
    p.terminate()
    exit(1)
else:
    print(f"\033[96mUsing device index: {device_index}\033[0m")
stream = p.open(format=pyaudio.paFloat32,
                channels=1,              # Mono input
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)

print("\033[96mAudio stream opened. Listening for real-time audio from ReSpeaker microphone...\033[0m")



# ---------------------- MFCC FEATURE EXTRACTION ----------------------
def extract_mfcc_features(audio, sr):
    """
    Compute MFCC features, average them over time, and normalize.
    Returns a fixed-length feature vector of length N_MFCC.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # shape (40, n_frames)
    mfcc = mfcc.T   # shape (n_frames, 40)  
    return (mfcc - np.mean(mfcc)) / np.std(mfcc)


# ---------------------- SLIDING WINDOW REAL-TIME PREDICTION LOOP ----------------------
try:
    audio_buffer = np.zeros(SAMPLE_RATE * AUDIO_LENGTH, dtype=np.float32)  # 2 seconds buffer
    step_size = int(0.5 * SAMPLE_RATE)  # 0.5-second step size

    while True:
        new_frames = []
        num_new_chunks = int(step_size / CHUNK)

        for _ in range(num_new_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            new_frames.append(np.frombuffer(data, dtype=np.float32))

        new_audio = np.concatenate(new_frames)

        # Update rolling buffer (append new and remove oldest)
        audio_buffer = np.roll(audio_buffer, -len(new_audio))
        audio_buffer[-len(new_audio):] = new_audio

        # Extract features from the current buffer
        features = extract_mfcc_features(audio_buffer, SAMPLE_RATE)
        features = features[np.newaxis, :, :]  


        # Make prediction
        prediction = model.predict(features, verbose=0)
        label = "Drone" if prediction[0][0] > 0.5 else "Background"
        print(f"Prediction: {label} (Probability: {prediction[0][0]:.3f})")

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

