import torch
import cv2
from ultralytics import YOLO

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("Ultralytics working")