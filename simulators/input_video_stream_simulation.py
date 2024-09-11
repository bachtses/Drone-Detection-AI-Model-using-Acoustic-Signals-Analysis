from flask import Flask, Response, redirect, url_for
import cv2
import numpy as np
import pyautogui

app = Flask(__name__)

def get_screen(region=None):
    while True:
        # Capture the screen
        img = pyautogui.screenshot(region=region)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Redirects to the video feed
    return redirect(url_for('video_feed'))

@app.route('/video_feed')
def video_feed():
    region = (100, 100, 500, 300)  # Adjust this to capture a specific region
    return Response(get_screen(region),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True, threaded=True)






