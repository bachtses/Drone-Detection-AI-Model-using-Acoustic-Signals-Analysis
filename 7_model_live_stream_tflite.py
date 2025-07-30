import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import pyaudio
import time
import threading
import matplotlib.pyplot as plt
import io
from matplotlib.patches import Rectangle

from flask import Flask, Response, render_template_string

# ---------------------- CONSTANTS ----------------------
SAMPLE_RATE = 24000
AUDIO_LENGTH = 2      # seconds
N_MFCC = 40
CHUNK = 1024
STEP_SIZE = int(0.2 * SAMPLE_RATE)  # sliding every x sec

# Shared audio buffer and lock
audio_buffer = np.zeros(SAMPLE_RATE * AUDIO_LENGTH, dtype=np.float32)
buffer_lock = threading.Lock()

# Latest prediction label
latest_label = "Background"
latest_probability = 0

# ---------------------- LOAD TFLITE MODEL ----------------------
try:
    interpreter = tf.lite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("\033[96mTFLite model loaded successfully.\033[0m")
except Exception as e:
    print("\033[91mFailed to load TFLite model:\033[0m", e)
    exit(1)

# ---------------------- AUDIO STREAM ----------------------
p = pyaudio.PyAudio()
try:
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("\033[96mAudio stream opened.\033[0m")
except Exception as e:
    print("\033[91mFailed to open audio stream:\033[0m", e)
    p.terminate()
    exit(1)

# ---------------------- FEATURE EXTRACTION ----------------------
def extract_mfcc_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    return (mfcc_mean - np.mean(mfcc_mean)) / np.std(mfcc_mean)

# ---------------------- PREDICTION THREAD ----------------------
def prediction_loop():
    global audio_buffer, latest_label, latest_probability
    while True:
        new_frames = []
        num_new_chunks = int(STEP_SIZE / CHUNK)
        for _ in range(num_new_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            new_frames.append(np.frombuffer(data, dtype=np.float32))

        new_audio = np.concatenate(new_frames)

        with buffer_lock:
            audio_buffer = np.roll(audio_buffer, -len(new_audio))
            audio_buffer[-len(new_audio):] = new_audio

        features = extract_mfcc_features(audio_buffer, SAMPLE_RATE)
        features = features.reshape(1, features.shape[0], 1).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        label = "Drone" if prediction[0][0] > 0.7 else "Background"
        latest_label = label
        latest_probability = prediction[0][0]
        print(f"Prediction: {label} (Probability: {prediction[0][0]:.3f})")

# ---------------------- FLASK SERVER ----------------------
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      margin: 0;
      padding: 0;
      background-color: black;
      overflow: hidden;
    }
    img {
      position: absolute;
      top: 0;
      left: 0;
      width: 100vw;
      height: auto;
      aspect-ratio: 16 / 9;
      object-fit: cover;
    }
  </style>
</head>
<body>
  <img src="{{ url_for('spectrogram') }}" />
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/spectrogram')
def spectrogram():
    return Response(generate_spectrogram(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_spectrogram():
    global audio_buffer, latest_label, latest_probability
    while True:
        with buffer_lock:
            y = audio_buffer.copy()

        fig, ax = plt.subplots(figsize=(10, 5.625))  # 16:9 aspect ratio
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=SAMPLE_RATE, x_axis='time', y_axis='log', ax=ax)
        ax.axis('off')

        if latest_label == "Drone":
            ax.text(0.5, 0.52, "DRONE DETECTED", transform=ax.transAxes,
                    fontsize=30, color='red', ha='center', va='center_baseline',
                    fontweight='bold')
            ax.text(0.5, 0.43, f"[ {latest_probability:.2f} ]", transform=ax.transAxes,
                    fontsize=20, color='red', ha='center', va='center_baseline',
                    fontweight='bold')
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             fill=False, color='red', linewidth=10)
            ax.add_patch(rect)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + buf.read() + b'\r\n')
        time.sleep(0.2)

# ---------------------- MAIN ----------------------
if __name__ == '__main__':
    prediction_thread = threading.Thread(target=prediction_loop)
    prediction_thread.daemon = True
    prediction_thread.start()

    print("\033[92mVisit http://localhost:5000 to see the live spectrogram.\033[0m")
    app.run(host='0.0.0.0', port=5000, threaded=True)
