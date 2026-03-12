import cv2
from ultralytics import YOLO
import time

# Load your custom trained model
model = YOLO('models/ambulance_best.pt')

class TrafficLane:
    def __init__(self, video_path, lane_id):
        self.cap = cv2.VideoCapture(video_path)
        self.lane_id = lane_id
        self.last_frame = None
        self.is_green = False
        self.ambulance_detected_start = None
        self.vehicle_count = 0

    def process(self):
        # If the light is RED, we return the last frame (the "pause" effect)
        if not self.is_green:
            return self.last_frame

        success, frame = self.cap.read()
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            success, frame = self.cap.read()

        # Run Detection
        results = model(frame, verbose=False)[0]
        self.vehicle_count = len(results.boxes)
        
        # Check for Ambulance Presence
        found_ambulance = False
        for box in results.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == 'ambulance':
                found_ambulance = True
                break
        
        # Logic for 3-second rule
        if found_ambulance:
            if self.ambulance_detected_start is None:
                self.ambulance_detected_start = time.time()
        else:
            self.ambulance_detected_start = None

        self.last_frame = frame
        return frame

    def check_emergency(self):
        if self.ambulance_detected_start:
            duration = time.time() - self.ambulance_detected_start
            return duration >= 3.0
        return False