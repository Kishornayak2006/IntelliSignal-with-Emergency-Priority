import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

# 1. Load Models with Performance Optimizations
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
car_model = YOLO('yolov8n.pt').to(device)
amb_model = YOLO('models/best.pt').to(device)

# Performance Tweak: Fuse layers and use Half-Precision (FP16) for RTX 4050
if device == 'cuda':
    car_model.fuse()
    amb_model.fuse()
    car_model.half()  # Butter smooth: FP16 math is 2x faster on Tensor Cores
    amb_model.half()

# 2. Setup Videos
video_paths = ["static/videos/lane1.mp4", "static/videos/lane2.mp4", 
               "static/videos/lane3.mp4", "static/videos/lane4.mp4"]
# Use FFMPEG backend for faster decoding
caps = [cv2.VideoCapture(v, cv2.CAP_FFMPEG) for v in video_paths]

# --- CONTROL VARIABLES ---
current_green_lane = 0
DEFAULT_GREEN = 15  
lane_timer = DEFAULT_GREEN
last_tick = time.time()

# Smoothness & Speed variables
RED_LANE_SPEED = 25  # Even slower for Red to save GPU for Green
loop_counter = 0

# Trackers
start_times = [None] * 4
amb_gone_timer = [0.0] * 4 
REQUIRED_SECONDS = 5
AMB_EXIT_DELAY = 3.0  
emergency_lane = None

def draw_signal(frame, status, timer_val):
    """Draws Traffic Light UI"""
    cv2.rectangle(frame, (530, 40), (620, 260), (30, 30, 30), -1)
    r_color = (0, 0, 255) if status == "RED" else (0, 0, 60)
    cv2.circle(frame, (575, 80), 22, r_color, -1)
    o_color = (0, 165, 255) if status == "ORANGE" else (0, 45, 60)
    cv2.circle(frame, (575, 150), 22, o_color, -1)
    g_color = (0, 255, 0) if status == "GREEN" else (0, 60, 0)
    cv2.circle(frame, (575, 220), 22, g_color, -1)
    cv2.putText(frame, f"{int(timer_val)}s", (555, 290), 1, 1.5, (255, 255, 255), 2)

def process_lane(frame, lane_idx, active_status):
    global start_times, emergency_lane, amb_gone_timer
    # High Smoothness: Resize once to small resolution for AI speed
    frame = cv2.resize(frame, (640, 360))
    current_now = time.time()

    # --- DETECTION ---
    # Performance Tweak: use stream=True and verbose=False
    car_res = car_model(frame, classes=[2, 3, 5, 7], conf=0.45, verbose=False, stream=False)[0]
    amb_res = amb_model(frame, conf=0.45, verbose=False, stream=False)[0]
    has_amb = len(amb_res.boxes) > 0

    # Draw detections
    for box in car_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for box in amb_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # --- EMERGENCY LOGIC ---
    if has_amb:
        amb_gone_timer[lane_idx] = 0 
        if start_times[lane_idx] is None: start_times[lane_idx] = current_now
        elapsed = int(current_now - start_times[lane_idx])
    else:
        if start_times[lane_idx] is not None:
            if amb_gone_timer[lane_idx] == 0: amb_gone_timer[lane_idx] = current_now
            if current_now - amb_gone_timer[lane_idx] > AMB_EXIT_DELAY:
                start_times[lane_idx], amb_gone_timer[lane_idx] = None, 0
                if emergency_lane == lane_idx: emergency_lane = None
                elapsed = 0
            else: elapsed = int(current_now - start_times[lane_idx])
        else: elapsed = 0

    if elapsed >= REQUIRED_SECONDS: emergency_lane = lane_idx

    # UI Elements
    draw_signal(frame, active_status, lane_timer if active_status != "RED" else 0)
    v_color = (0, 0, 255) if elapsed >= REQUIRED_SECONDS else (0, 255, 255)
    cv2.rectangle(frame, (20, 310), (330, 345), (0,0,0), -1)
    status_msg = "PRIORITY GRANTED" if emergency_lane == lane_idx else f"VERIFYING AMBULANCE: {min(elapsed, 5)}/5s"
    cv2.putText(frame, status_msg, (30, 335), 1, 1.1, v_color, 2)
    return frame

# --- INITIALIZE ---
frames = [None] * 4
for i in range(4):
    ret, img = caps[i].read()
    frames[i] = img if ret else np.zeros((360, 640, 3), dtype=np.uint8)

# --- MAIN LOOP ---
while True:
    current_now = time.time()
    loop_counter += 1
    active_lane = emergency_lane if emergency_lane is not None else current_green_lane
    
    if current_now - last_tick >= 1.0:
        lane_timer -= 1
        last_tick = current_now

    # Cycle Logic
    active_status = "GREEN"
    if emergency_lane is None:
        if 0 < lane_timer <= 3: active_status = "ORANGE"
        elif lane_timer <= 0:
            current_green_lane = (current_green_lane + 1) % 4
            lane_timer = DEFAULT_GREEN

    processed_results = []
    for i in range(4):
        
        if (i == active_lane) or (loop_counter % RED_LANE_SPEED == 0):
            ret, img = caps[i].read()
            if not ret:
                caps[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, img = caps[i].read()
            if ret: frames[i] = img

        lane_signal = active_status if i == active_lane else "RED"
        p_frame = process_lane(frames[i].copy(), i, lane_signal)
        processed_results.append(p_frame)

    # Rendering
    top = np.hstack((processed_results[0], processed_results[1]))
    bottom = np.hstack((processed_results[2], processed_results[3]))
    grid = np.vstack((top, bottom))
    
    cv2.imshow("IntelliSignal with EMERGENCY priority", np.ascontiguousarray(grid))
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

for c in caps: c.release()
cv2.destroyAllWindows()