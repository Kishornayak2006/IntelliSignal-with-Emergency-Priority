import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load your specialists
car_model = YOLO('yolov8n.pt')
amb_model = YOLO('models/best.pt')

# 2. Open all 4 lanes
video_sources = [
    "static/videos/lane1.mp4",
    "static/videos/lane2.mp4",
    "static/videos/lane3.mp4",
    "static/videos/lane4.mp4"
]
caps = [cv2.VideoCapture(src) for src in video_sources]

def get_analyzed_frame(cap, lane_num):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video if it ends
        ret, frame = cap.read()
    
    # Resize for the grid (640x360 is perfect for 1080p screens)
    frame = cv2.resize(frame, (640, 360))
    
    # Run Detections
    car_results = car_model(frame, classes=[2, 3, 5, 7], verbose=False)[0]
    amb_results = amb_model(frame, verbose=False)[0]
    
    car_count = len(car_results.boxes)
    has_ambulance = len(amb_results.boxes) > 0
    
    # Logic for Signal Color
    # Rule: If Ambulance is present, Green. Otherwise, if high traffic, Red.
    signal_color = (0, 255, 0) if has_ambulance else (0, 0, 255)
    status_text = "EMERGENCY" if has_ambulance else f"Cars: {car_count}"
    
    # Draw Signal Light (Circle)
    cv2.circle(frame, (50, 50), 20, signal_color, -1)
    cv2.putText(frame, status_text, (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"LANE {lane_num}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    return frame

while True:
    # Process all 4 lanes
    l1 = get_analyzed_frame(caps[0], 1)
    l2 = get_analyzed_frame(caps[1], 2)
    l3 = get_analyzed_frame(caps[2], 3)
    l4 = get_analyzed_frame(caps[3], 4)
    
    # Combine into 2x2 Grid
    top_row = np.hstack((l1, l2))
    bottom_row = np.hstack((l3, l4))
    dashboard = np.vstack((top_row, bottom_row))
    
    cv2.imshow("IntelliSignal AI - 4 Lane Priority Dashboard", dashboard)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps: cap.release()
cv2.destroyAllWindows()