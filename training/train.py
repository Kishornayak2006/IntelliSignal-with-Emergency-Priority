import os
from ultralytics import YOLO

# 1. RTX 40-series DLL Fix (Prevents the 'Procedure Not Found' error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def start_training():
    # 2. Load the base model
    model = YOLO('yolov8n.pt') 

    # 3. Start the GPU training
    # Change the 'data' path if your yaml file is in a different location
    results = model.train(
        data="E:/IntelliSignal AI- with Emergency priority/dataset/ambulance-dataset/data.yaml", 
        epochs=50,       # 50 passes over the data
        imgsz=640,      # Standard image size
        batch=16,       # How many images the GPU processes at once
        device=0,       # FORCE GPU (RTX 4050)
        project='models',
        name='ambulance_priority'
    )

if __name__ == "__main__":
    start_training()