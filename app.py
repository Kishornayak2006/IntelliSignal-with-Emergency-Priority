import cv2
import time
import os
from flask import Flask, render_template, Response
from ultralytics import YOLO
from threading import Thread

# 1. PREVENT THE OMP ERROR
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = Flask(__name__)

# Load Model: Standard YOLO for vehicles
# You will replace 'yolov8n.pt' with 'models/ambulance_priority/weights/best.pt' later
model = YOLO('yolov8n.pt') 

class Lane:
    def __init__(self, video_path, lane_id):
        self.cap = cv2.VideoCapture(video_path)
        self.lane_id = lane_id
        self.is_green = False
        self.vehicle_count = 0
        self.ambulance_timer = 0
        self.last_frame = None

    def process(self):
        # If RED: Freeze the video at the last frame
        if not self.is_green:
            if self.last_frame is not None:
                return self.annotate(self.last_frame.copy(), "RED")
            return None

        # If GREEN: Read and play
        success, frame = self.cap.read()
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            success, frame = self.cap.read()

        results = model(frame, verbose=False)[0]
        
        # Count vehicles (Classes 2,3,5,7 are car, bike, bus, truck)
        self.vehicle_count = len([b for b in results.boxes if int(b.cls) in [2,3,5,7]])
        
        # Check for Ambulance (If you train it as class 0)
        has_ambulance = any(int(b.cls) == 0 for b in results.boxes)
        if has_ambulance:
            self.ambulance_timer += 0.1 # Approx increment per loop
        else:
            self.ambulance_timer = 0

        self.last_frame = frame.copy()
        return self.annotate(frame, "GREEN")

    def annotate(self, frame, status):
        color = (0, 255, 0) if status == "GREEN" else (0, 0, 255)
        cv2.putText(frame, f"LANE {self.lane_id}: {status}", (20, 50), 1, 2, color, 3)
        cv2.putText(frame, f"Cars: {self.vehicle_count}", (20, 100), 1, 1.5, (255,255,255), 2)
        return frame

# Initialize Lanes
video_files = ['static/videos/lane1.mp4', 'static/videos/lane2.mp4', 'static/videos/lane3.mp4', 'static/videos/lane4.mp4']
lanes = [Lane(v, i+1) for i, v in enumerate(video_files)]
current_idx = 0
lanes[0].is_green = True
start_time = time.time()

def controller():
    global current_idx, start_time
    while True:
        time.sleep(0.1)
        elapsed = time.time() - start_time
        
        # EMERGENCY RULE: Check all lanes for ambulance > 3s
        for i, lane in enumerate(lanes):
            if lane.ambulance_timer >= 3.0 and current_idx != i:
                switch(i)
                break
        
        # DENSITY RULE: Jump if < 3 vehicles (after 5s buffer)
        if lanes[current_idx].vehicle_count < 3 and elapsed > 5:
            switch((current_idx + 1) % 4)

        # TIMER RULE: 30s
        if elapsed >= 30:
            switch((current_idx + 1) % 4)

def switch(new_idx):
    global current_idx, start_time
    for l in lanes: l.is_green = False
    lanes[new_idx].is_green = True
    current_idx = new_idx
    start_time = time.time()

# Flask Routes
@app.route('/')
def index(): return render_template('index.html')

def gen(id):
    while True:
        frame = lanes[id].process()
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/video_feed/<int:id>')
def video_feed(id): return Response(gen(id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    Thread(target=controller, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)