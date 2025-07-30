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
from matplotlib.patches import Rectangle  # Put this at the top with imports
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend (no GUI)
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
confidence_history = []
CONF_HISTORY_LEN = 50
 

# ---------------------- LOAD MODEL ----------------------
try:
    model = tf.keras.models.load_model('model.h5')
    print("\033[96mModel loaded successfully.\033[0m")
except Exception as e:
    print("\033[91mFailed to load the model:\033[0m", e)
    exit(1)


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
print("Audio stream opened. Listening for real-time audio from ReSpeaker microphone...")


# ---------------------- FEATURE EXTRACTION ----------------------
def extract_mfcc_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # shape (40, n_frames)
    mfcc = mfcc.T   # shape (n_frames, 40)  
    return (mfcc - np.mean(mfcc)) / np.std(mfcc)

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
        features = features[np.newaxis, :, :]  
        prediction = model.predict(features, verbose=0)
        label = "Drone" if prediction[0][0] > 0.7 else "Background"
        latest_label = label
        latest_probability = prediction[0][0]
        confidence_history.append(latest_probability)
        if len(confidence_history) > CONF_HISTORY_LEN:
            confidence_history.pop(0)

        print(f"Prediction: {label} (Probability: {prediction[0][0]:.3f})")

# ---------------------- FLASK SERVER ----------------------
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      background-color: black;
      overflow: hidden;
      height: 100%;
      width: 100%;
    }
    #wrapper {
      position: absolute;
      top: 0;
      left: 0;
    }
    img {
      display: block;
      margin: 0;
      padding: 0;
      width: auto;
      height: auto;
    }
  </style>
</head>
<body>
  <div id="wrapper">
    <img src="{{ url_for('spectrogram') }}">
  </div>
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
    global audio_buffer, latest_label, latest_probability, confidence_history

    while True:
        with buffer_lock:
            y = audio_buffer.copy()
            conf_hist = confidence_history.copy()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 5.625) ,
            gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.04}  
        )

        black_color = "#0a0a0a"
        ax1.set_facecolor(black_color)
        dark_purple = '#030313'
        ax2.set_facecolor(dark_purple)
        fig.patch.set_facecolor('white')

        # Top: Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=SAMPLE_RATE, x_axis='time', y_axis='log', ax=ax1)
        ax1.set_ylabel('')  
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', bottom=False, labelbottom=False)
        ax1.set_yticks([])
        ax1.text(0.01, 0.1, '500 Hz', color='white', ha='left', va='center', fontsize=10, transform=ax1.transAxes)
        ax1.text(0.01, 0.3, '1k Hz', color='white', ha='left', va='center', fontsize=10, transform=ax1.transAxes)
        ax1.text(0.01, 0.55, '2k Hz', color='white', ha='left', va='center', fontsize=10, transform=ax1.transAxes)
        ax1.text(0.01, 0.8, '4k Hz', color='white', ha='left', va='center', fontsize=10, transform=ax1.transAxes)

        if latest_label == "Drone":
            ax1.text(0.5, 0.52, "DRONE DETECTED", transform=ax1.transAxes,
                     fontsize=30, color='red', ha='center', va='center_baseline', fontweight='bold')
            ax1.text(0.5, 0.43, f"[ {latest_probability:.2f} ]", transform=ax1.transAxes,
                     fontsize=20, color='red', ha='center', va='center_baseline', fontweight='bold')
            rect = Rectangle((0, 0), 1, 1, transform=ax1.transAxes,
                             fill=False, color='red', linewidth=10)
            ax1.add_patch(rect)

        # Bottom: Confidence plot
        conf_array = np.array(conf_hist)
        x_vals = np.arange(len(conf_array))
        color_for_dot = "red"
        ax2.plot(x_vals, conf_array, color= color_for_dot, linewidth=2)
        light_orange = '#ff9465'
        ax2.fill_between(x_vals, 0, conf_array, color= light_orange, alpha=0.7)
        ax2.set_ylim([0, 1])
        ax2.set_xlim([0, CONF_HISTORY_LEN - 1])
        ax2.set_yticks([])

        # Add custom labels at specific positions (closer together)
        ax2.text(0.35, 0.15, 'Background', color='white', ha='left', va='center', fontsize=10)
        ax2.text(0.35, 0.85, 'Drone', color='white', ha='left', va='center', fontsize=10)

        ax2.tick_params(axis='x', bottom=False, labelbottom=False)
        ax2.grid(True, linestyle='--', alpha=0.0, color='white')

        for spine in ax2.spines.values():
            spine.set_visible(False)

        buf = io.BytesIO()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + buf.read() + b'\r\n')
        time.sleep(0.1)




# ---------------------- MAIN ----------------------
if __name__ == '__main__':
    prediction_thread = threading.Thread(target=prediction_loop)
    prediction_thread.daemon = True
    prediction_thread.start()

    print("\033[92mVisit http://localhost:5000 to see the live spectrogram.\033[0m")
    app.run(host='0.0.0.0', port=5000, threaded=True)






