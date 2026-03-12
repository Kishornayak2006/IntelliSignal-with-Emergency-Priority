import cv2
import numpy as np
from ultralytics import YOLO
import time # Added to track real clock time

# 1. Load Models
car_model = YOLO('yolov8n.pt')          
amb_model = YOLO('models/best.pt') 

# 2. Setup
video_paths = ["static/videos/lane1.mp4", "static/videos/lane2.mp4", 
               "static/videos/lane3.mp4", "static/videos/lane4.mp4"]
caps = [cv2.VideoCapture(v) for v in video_paths]

REQUIRED_SECONDS = 5

# --- NEW TRACKERS ---
# We store the 'start time' when an ambulance is first seen
start_times = [None, None, None, None] 
# Grace period to prevent the timer from resetting if detection blinks for a split second
last_seen_times = [0, 0, 0, 0] 

def process_lane(frame, lane_idx):
    global start_times, last_seen_times
    lane_id = lane_idx + 1
    frame = cv2.resize(frame, (640, 360))
    current_now = time.time()
    
    # --- DETECTION (Original Logic) ---
    car_res = car_model(frame, classes=[2, 3, 5, 7], conf=0.5, verbose=False)[0]
    amb_res = amb_model(frame, conf=0.5, verbose=False)[0]
    
    # --- DRAW LIVE BOXES ---
    for box in car_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    has_amb = len(amb_res.boxes) > 0
    for box in amb_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, "AMBULANCE", (x1, y1-10), 1, 1, (0, 0, 255), 2)

    # --- UPDATED TIMER LOGIC (REAL CLOCK TIME) ---
    if has_amb:
        last_seen_times[lane_idx] = current_now
        # If this is the first time we see it, start the clock
        if start_times[lane_idx] is None:
            start_times[lane_idx] = current_now
        
        elapsed = int(current_now - start_times[lane_idx])
    else:
        # If ambulance is gone for more than 1.5 seconds, reset timer
        if current_now - last_seen_times[lane_idx] > 1.5:
            start_times[lane_idx] = None
            elapsed = 0
        else:
            # Keep the time if it's just a quick blink
            elapsed = int(current_now - start_times[lane_idx]) if start_times[lane_idx] else 0

    current_seconds = min(elapsed, REQUIRED_SECONDS)
    is_emergency = current_seconds >= REQUIRED_SECONDS
    
    # --- UI OVERLAY ---
    overlay_color = (0, 0, 255) if is_emergency else (0, 255, 0)
    status_text = "EMERGENCY ACTIVE" if is_emergency else f"Verifying: {current_seconds}/{REQUIRED_SECONDS}s"
    
    cv2.rectangle(frame, (0,0), (640, 50), (0,0,0), -1)
    cv2.putText(frame, f"LANE {lane_id} | {status_text}", (20, 35), 1, 1.2, overlay_color, 2)
    
    return frame

while True:
    frames = []
    for i, cap in enumerate(caps):
        ret, f = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, f = cap.read()
        frames.append(process_lane(f, i))

    # Stack into 2x2 Grid
    top = np.hstack((frames[0], frames[1]))
    bot = np.hstack((frames[2], frames[3]))
    grid = np.vstack((top, bot))

    # Fix for memory layout (prevents OpenCV errors)
    grid = np.ascontiguousarray(grid, dtype=np.uint8)

    cv2.imshow("IntelliSignal AI - 4-Lane Emergency Dashboard", grid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps: cap.release()
cv2.destroyAllWindows()