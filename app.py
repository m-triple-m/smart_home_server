from flask import Flask, render_template, Response, stream_with_context, request, jsonify
import requests
import time
import os
import face_engine
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Configuration ---
ESP32_CAM_IP = "192.168.18.241" 
STREAM_URL = f"http://{ESP32_CAM_IP}:81/stream"
CAPTURE_URL = f"http://{ESP32_CAM_IP}/capture"
UPLOAD_FOLDER = 'static/uploads'
TEMP_FOLDER = 'static/temp'

for folder in [UPLOAD_FOLDER, TEMP_FOLDER, face_engine.DEFAULT_FACES_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def gen_frames():
    """Generator function that fetches MJPEG stream from ESP32 with optimized buffering."""
    while True:
        try:
            with requests.get(STREAM_URL, stream=True, timeout=5) as res:
                if res.status_code != 200:
                    time.sleep(2)
                    continue
                for chunk in res.iter_content(chunk_size=65536):
                    if chunk:
                        yield chunk
        except Exception as e:
            time.sleep(2)

def capture_frame():
    """Captures a single frame from the ESP32-CAM."""
    try:
        # Try the /capture endpoint first (usually higher res)
        response = requests.get(CAPTURE_URL, timeout=5)
        if response.status_code == 200:
            temp_path = os.path.join(TEMP_FOLDER, "current_frame.jpg")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            return temp_path
    except:
        pass
    
    # Fallback: Try to grab one frame from the stream
    try:
        with requests.get(STREAM_URL, stream=True, timeout=5) as res:
            byte_data = b''
            for chunk in res.iter_content(chunk_size=65536):
                byte_data += chunk
                a = byte_data.find(b'\xff\xd8')
                b = byte_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = byte_data[a:b+2]
                    temp_path = os.path.join(TEMP_FOLDER, "current_frame.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(jpg)
                    return temp_path
    except Exception as e:
        print(f"Capture failed: {e}")
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(gen_frames()), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognize', methods=['POST'])
def recognize():
    frame_path = capture_frame()
    if not frame_path:
        return jsonify({"status": "error", "message": "Could not capture frame from camera."})
    
    result = face_engine.match_face(frame_path)
    return jsonify(result)

@app.route('/add_face', methods=['POST'])
def add_face():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"status": "error", "message": "Missing image or name."})
    
    file = request.files['image']
    name = request.form['name']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."})
    
    filename = secure_filename(file.filename)
    temp_path = os.path.join(TEMP_FOLDER, filename)
    file.save(temp_path)
    
    result = face_engine.add_new_face(temp_path, name)
    return jsonify(result)

@app.route('/list_faces')
def list_faces():
    faces = face_engine.get_all_faces()
    return jsonify({"status": "success", "faces": faces})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
