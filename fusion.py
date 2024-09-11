from flask import Flask, Response
import requests
import cv2
import numpy as np
from keras.models import load_model
import time
import threading
import logging
import socket
import queue
import json
from flask import request


# Constants
LABELS = [1, 0] #["Drone", "No Drone"]
IMG_WIDTH = 256
IMG_HEIGHT = 117

app = Flask(__name__)


###############################################################################################
################                       Output Json Sender                      ################
###############################################################################################
def send_json_data(url, payload):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    return response.status_code, response.text

def send_json_to_platform(confidence_fusion, confidence_gfai, confidence_respeaker, angle_respeaker, wav_url_respeaker, detected):
    # send json to platform               
    url = "https://api.safekibo.gr/api/import" #"http://160.40.55.51:8000/api/import" 
    payload = {
        "password": "70aac099-572e-4bb7-80a0-0e9493cc84e0",
        "station_id": 1,
        "sensor_id": 7,
        "type": "acoustic_fusion",
        "data": [
            { 
            "acoustic_gfai": {
                "station_id": 1,
                "sensor_id": 2,
                "detected": detected,
                "confidence": confidence_gfai
            },
            "acoustic_raspberry": {
                "station_id": 1,
                "sensor_id": 6,
                "detected": detected,
                "confidence": confidence_respeaker,
                "angle": angle_respeaker,
                "wav_url": wav_url_respeaker
            },
            "objectId": 1,
            "detected": detected,
            "confidence": confidence_fusion,
            "unknown": True
            }
        ]
    }
    status_code, response_text = send_json_data(url, payload)
    print(f"\033[94m[Json Sender] Sending... Detected: {detected} Response Status: {status_code} \033[0m")
    

# Buffers 
MAX_QUEUE_SIZE = 20  # Maximum items in the queue
logs_gfai = queue.Queue()
logs_respeaker = queue.Queue()
def add_to_queue(q, item):
    if q.qsize() >= MAX_QUEUE_SIZE:
        q.get()  # Remove the oldest item
    q.put(item)

###############################################################################################
################         Input Socket listener for GFAI Unimodal Logs          ################
###############################################################################################
def socket_listener_gfai_unimodal(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print("[Input Socket GFAI Unimodal] Server started. Waiting for connection...")
        connection, addr = server_socket.accept()
        with connection:
            print(f"[Input Socket GFAI Unimodal] Connected by {addr}")
            try:
                while True:
                    data = connection.recv(1024)
                    if not data:
                        break
                    data_str = data.decode()
                    label_gfai, confidence_gfai, timestamp_gfai = data_str.split(',')
                    print(f"[Input Socket GFAI Unimodal]         Label: {label_gfai} Confidence: {confidence_gfai} Timestamp: {timestamp_gfai} ")
                    add_to_queue(logs_gfai, (label_gfai, confidence_gfai, timestamp_gfai))

            except ConnectionResetError:
                print("\033[91m[Input Socket GFAI Unimodal] Connection lost. \033[0m")
            finally:
                print("\033[91m[Input Socket GFAI Unimodal] Connection closed. \033[0m")
              



###############################################################################################
################       Input Socket listener for Respeaker Unimodal Logs       ################
###############################################################################################
def socket_listener_respeaker_unimodal(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print("[Input Socket Respeaker Unimodal] Server started. Waiting for connection...")
        connection, addr = server_socket.accept()
        with connection:
            print(f"[Input Socket Respeaker Unimodal] Connected by {addr}")
            try:
                while True:
                    data = connection.recv(1024)
                    if not data:
                        break
                    data_str = data.decode()
                    timestamp_respeaker, label_respeaker, confidence_respeaker, angle_respeaker, wav_url_respeaker = data_str.split(',')
                    print(f"[Input Socket Respeaker Unimodal]    Label: {label_respeaker} Confidence: {confidence_respeaker} Timestamp: {timestamp_respeaker} Angle: {angle_respeaker} Wav: {wav_url_respeaker}")
                    add_to_queue(logs_respeaker, (label_respeaker, confidence_respeaker, timestamp_respeaker, angle_respeaker, wav_url_respeaker))  
            except ConnectionResetError:
                print("[Input Socket Respeaker Unimodal] Connection lost.")
            finally:
                print("[Input Socket Respeaker Unimodal] Connection closed.")



###############################################################################################
################               Fusion / Buffers synchronization                ################
###############################################################################################
last_detected_state = False
last_detected_wav = ""

def synchronize_buffers_and_fusion_result():
    global last_detected_state, last_detected_wav
    detected = 0  
    while not logs_gfai.empty() and not logs_respeaker.empty():
        gfai_current_log = logs_gfai.queue[0]
        respeaker_current_log = logs_respeaker.queue[0]
        timestamp_gfai = float(gfai_current_log[2])
        timestamp_respeaker = float(respeaker_current_log[2])
        if abs(timestamp_gfai - timestamp_respeaker) < 1:
            logs_gfai.get()
            logs_respeaker.get()
            print(f"\033[93m[Fusion / Buffers Synchronizing] [GFAI Logs] {gfai_current_log} [Respeaker Logs] {respeaker_current_log} \033[0m")
            label_gfai = float(gfai_current_log[0])
            confidence_gfai = float(gfai_current_log[1])
            label_respeaker = float(respeaker_current_log[0])
            confidence_respeaker = float(respeaker_current_log[1])
            angle_respeaker = float(respeaker_current_log[3])
            wav_url_respeaker = str(respeaker_current_log[4])
            
            if (label_gfai == 1 and label_respeaker == 1) or (label_gfai == 1 and confidence_gfai >= 0.75) or (label_respeaker == 1 and confidence_respeaker >= 0.75):
                confidence_fusion = max(confidence_gfai, confidence_respeaker)
                print(f"\033[92m[Fusion] Drone Detected! Confidence: {confidence_fusion} \033[0m")
                #sending json to platform
                detected = 1
                last_detected_state = True
                last_detected_wav = wav_url_respeaker
                send_json_to_platform(confidence_fusion, confidence_gfai, confidence_respeaker, angle_respeaker, wav_url_respeaker, detected)
                #return message to display
                message_to_display1 = f'Drone Detected '
                message_to_display2 = f'with confidence: {confidence_fusion*100}%'

                return True, message_to_display1, message_to_display2
        elif timestamp_gfai < timestamp_respeaker:
            logs_gfai.get() # If GFAI log is older, remove it
            return False, None, None
        else:
            logs_respeaker.get() # If Respeaker log is older, remove it
            return False, None, None
        
    if last_detected_state and detected == 0:
        # Reset last detection state and send detected=0 to notify that no drone is currently detected
        send_json_to_platform(0, 0, 0, 0, last_detected_wav, detected)  # Modify parameters as needed
        last_detected_state = False
        return False, 'No drone detected.', 'No drone detected.'

    return False, None, None


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
        logging.error("\033[91m[Input GFAI Video Stream] Error opening video stream or file \033[0m")
        return
    last_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning("\033[91m[Input GFAI Video Stream] Frame not received. Attempting to reconnect...\033[0m")
            time.sleep(10)  # wait before reconnecting
            cap = cv2.VideoCapture(INPUT_STREAM_IP)
            continue
        current_time = time.time()
        if current_time - last_time >= interval:
            last_time = current_time

            fusion_result, message_to_display1, message_to_display2 = synchronize_buffers_and_fusion_result()
        
            if fusion_result == True: # For each frame check the fusion of the models
                # Adding notification
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 14)

                text_size1 = cv2.getTextSize(message_to_display1, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 4)[0]
                text_x1 = (frame.shape[1] - text_size1[0]) // 2
                text_y1 = (frame.shape[0] + text_size1[1]) // 2 - 20  # Adjust the Y position upwards for message 1
                cv2.putText(frame, message_to_display1, (text_x1, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 4)

                text_size2 = cv2.getTextSize(message_to_display2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 4)[0]
                text_x2 = (frame.shape[1] - text_size2[0]) // 2
                text_y2 = text_y1 + text_size1[1] + 20  # Adjust the Y position for message 2 (below message 1)
                cv2.putText(frame, message_to_display2, (text_x2, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 4)

            with condition:
                processed_frame = frame
                condition.notify()
    cap.release()



###############################################################################################
################                Output video stream to platform                ################
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


@app.route('/status', methods=['GET'])
def status():
    return {"message": "up"}, 200


   


if __name__ == "__main__":
    
    # Starting the video stream thread
    INPUT_STREAM_IP = 'http://160.40.52.69:5010' # alex: 'http://160.40.50.132:5010/video_feed' # IP of the PC that streams the GFAI spectrogram stream
    video_thread = threading.Thread(target=video_stream_prediction, args=(INPUT_STREAM_IP,))
    video_thread.daemon = True
    video_thread.start()
    
    # Starting the socket listener thread for gfai logs  
    HOST_gfai = '0.0.0.0' # IP of the PC that runs the gfai unimodal component
    PORT_gfai = 49150
    socket_thread = threading.Thread(target=socket_listener_gfai_unimodal, args=(HOST_gfai, PORT_gfai))
    socket_thread.daemon = True
    socket_thread.start()

    # Starting the socket listener thread for respeaker logs
    HOST_respeaker = '0.0.0.0' 
    PORT_respeaker = 49151
    socket_thread = threading.Thread(target=socket_listener_respeaker_unimodal, args=(HOST_respeaker, PORT_respeaker))
    socket_thread.daemon = True
    socket_thread.start()

    # Starting the fusion output video stream thread
    app.run(host='0.0.0.0', port=8444, debug=False, threaded=True, ssl_context=('ssl/cert.pem', 'ssl/key.pem'))
    



   
    