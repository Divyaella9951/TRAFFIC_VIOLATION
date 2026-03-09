from flask import Flask, render_template, jsonify, Response
import cv2
from master_traffic_system import run_detection

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def generate_frames():

    while True:
        success, frame = camera.read()
        if not success:
            continue
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route("/detect")
def detect():
    data = run_detection()
    print("DATA SENT TO DASHBOARD:", data)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)