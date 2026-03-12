try:
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    import torch
    import torchvision
    from ultralytics import YOLO
    print("--- Environment Check ---")
    print(f"Torch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Check failed: {e}")