import socket
import time
import random

HOST = '127.0.0.1' #server's IP 160.40.48.63  # Server address (localhost if testing on the same machine)
PORT = 49151  # Port number where the server is listening

def send_data(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect((host, port))
            print("Connected to server. \n")
        except ConnectionRefusedError:
            print("Connection failed. Is the server running? \n")
            return
        try:
            while True:
                # Data simulation
                timestamp = int(time.time())  
                timestamp = timestamp
                label = 0 #random.choice([0, 1])  
                confidence = round(random.uniform(0.5, 1.0), 2) 
                angle = random.choice([50])
                wav_url = "https://160.40.55.75:5000/files/recording_2024-08-26_12-32-32.wav"
                data = f"{timestamp},{label},{confidence},{angle},{wav_url}"
                print(f"Label: {label}  Confidence: {confidence}  Timestamp: {timestamp}  Angle: {angle} Wav: {wav_url}")
                client_socket.sendall(data.encode())
                time.sleep(1)
        except KeyboardInterrupt:
            print("Disconnected from server.")
        finally:
            client_socket.close()

if __name__ == "__main__":
    send_data(HOST, PORT)
