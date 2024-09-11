from flask import Flask, Response
import cv2
import numpy as np
from keras.models import load_model
import time
import threading
import logging
import socket 
from queue import Queue

# INPUT: video stream from GFAI URL
# OUTPUT: video stream to platform URL
# OUTPUT: json logs to fusion component via socket

# Constants
LABELS = [1, 0] #["Drone", "No Drone"]
IMG_WIDTH = 256
IMG_HEIGHT = 117

app = Flask(__name__)

# Load model once
try:
    model = load_model('model.h5')
except Exception as e:
    logging.error(f"[Model Load] Failed to load the model: {e}")

###############################################################################################
################       Output sending logs to fusion component via socket      ################
###############################################################################################
logs_queue = Queue()

def send_data(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect((host, port))
            print("\033[92m[Logs Output Socket] Connected to fusion component. \033[0m")
        except ConnectionRefusedError:
            print("\033[91m[Logs Output Socket] Connection failed. Is the fusion component running? \033[0m")
            return
        try:
            while True:
                logs = logs_queue.get()
                client_socket.sendall(logs.encode())
                logs_queue.task_done()
                print("\033[94m[Logs Output Socket] Sending... \033[0m")
        except KeyboardInterrupt:
            print("[Logs Output Socket] Disconnected from fusion component.")
        finally:
            client_socket.close()
        

###############################################################################################
################                 Input video stream from GFAI                  ################
###############################################################################################
frame_lock = threading.Lock()
condition = threading.Condition(frame_lock)
processed_frame = None


def video_stream_prediction(INPUT_STREAM_IP, interval=0.5):
    global processed_frame
    cap = cv2.VideoCapture(INPUT_STREAM_IP)
    if not cap.isOpened():
        logging.error("\033[91m[GFAI Input Stream] Error opening video stream or file \033[0m")
        return
    last_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning("\033[91m[GFAI Input Stream] Frame not received. Attempting to reconnect...\033[0m")
            time.sleep(10)  # wait before reconnecting
            cap = cv2.VideoCapture(INPUT_STREAM_IP)
            continue
        current_time = time.time()
        if current_time - last_time >= interval:
            last_time = current_time
            frame = perform_prediction(frame)
            with condition:
                processed_frame = frame
                condition.notify()  
    cap.release()



###############################################################################################
################                Run model and perform prediction               ################
###############################################################################################
def perform_prediction(frame):
    resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    normalized_frame = resized_frame / 255.0
    frame_to_predict = np.expand_dims(normalized_frame, axis=0)
    prediction = model.predict(frame_to_predict, verbose=0)
    predicted_label = np.argmax(prediction)
    confidence = float(prediction[0][predicted_label])
    confidence = round(confidence, 2)
    current_time = int(time.time())
    print(f"[GFAI Unimodal] Label: {LABELS[predicted_label]} Confidence: {confidence} Timestamp: {current_time}")
    # Send logs via socket
    logs = f"{LABELS[predicted_label]},{confidence},{current_time}"
    logs_queue.put(logs)
    if LABELS[predicted_label] == 1:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 14)

        text_size1 = cv2.getTextSize(f'Drone Detected', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)[0]
        text_x1 = (frame.shape[1] - text_size1[0]) // 2
        text_y1 = (frame.shape[0] + text_size1[1]) // 2 - 20  
        cv2.putText(frame, f'Drone Detected', (text_x1, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  

        text_size2 = cv2.getTextSize(f'with confidence: {confidence*100}%', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)[0]
        text_x2 = (frame.shape[1] - text_size2[0]) // 2
        text_y2 = text_y1 + text_size1[1] + 10 
        cv2.putText(frame, f'with confidence: {confidence*100}%', (text_x2, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
       


    return frame  # Return the modified frame



###############################################################################################
################                   Output video stream to IP                   ################
###############################################################################################
def generate_video():
    global processed_frame
    while True:
        with condition:
            condition.wait()  # Wait until a frame is ready
            if processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return "Video stream processing is running in the background. Visit /video_feed to view the stream."



if __name__ == "__main__":
    # Starting the video stream thread
    INPUT_STREAM_IP = 'http://160.40.52.69:5010' #alex: 'http://160.40.50.132:5010/video_feed'  # IP of the PC that streams the GFAI spectrogram stream
    video_thread = threading.Thread(target=video_stream_prediction, args=(INPUT_STREAM_IP,))
    video_thread.daemon = True
    video_thread.start()

    # Starting the socket for sendind logs to fusion component
    HOST_gfai = '127.0.0.1' #IP of the PC that runs the fusion component
    PORT_gfai = 49150
    socket_thread = threading.Thread(target=send_data, args=(HOST_gfai, PORT_gfai))
    socket_thread.daemon = True
    socket_thread.start()

    # Starting the unimodal output video stream thread
    app.run(host='0.0.0.0', port=8443, debug=False, threaded=True, ssl_context=('ssl/cert.pem', 'ssl/key.pem'))
