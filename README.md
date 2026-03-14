# 🚦 IntelliSignal AI: Emergency-Priority Traffic Management
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-red)](https://ultralytics.com/)
[![CUDA](https://img.shields.io/badge/Hardware-RTX%204050%20Optimized-orange?logo=nvidia)](https://developer.nvidia.com/cuda-zone)

**IntelliSignal AI** is an intelligent traffic control system that uses Computer Vision to solve the "Golden Hour" problem. It autonomously identifies ambulances and grants them priority passage through intersections, potentially saving lives by reducing emergency response times.

---

## 📺 Project Demonstration
Capture the system in action. This demo highlights the real-time detection and seamless signal transition.

[![IntelliSignal Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_LINK)
*Click the image above to watch the full system demonstration.*

---

## 🛠️ Key Characteristics & Innovation

### 1. Emergency Priority Latching
The system doesn't just "see" an ambulance; it **latches** the priority. Once an ambulance is verified for 5 seconds, the signal turns green and stays green until the vehicle has passed, regardless of the standard timer.

### 2. 10-Second Clearance Buffer
To prevent accidents, the system maintains a **10-second grace period** after the ambulance leaves the camera's view. This ensures the emergency vehicle has fully cleared the intersection before cross-traffic resumes.

### 3. Hardware-Aware Performance (RTX 4050)
* **FP16 Half-Precision:** Optimized for NVIDIA Tensor Cores, providing 2x faster inference.
* **Dynamic FPS Logic:** The active "Green" lane runs at 100% frame rate for high precision, while "Red" lanes scale down to 0.05x speed to conserve GPU resources without losing detection capability.

---

## 📐 System Architecture
| 4-Lane Monitoring | Ambulance Verification | Signal Clearance |
| :---: | :---: | :---: |
| ![Grid View](assets/grid_view.png) | ![Detection](assets/detection_closeup.png) | ![Timer](assets/clearance_timer.png) |
| *Simultaneous YOLOv8 tracking on 4 independent streams.* | *Dual-model logic for high-accuracy classification.* | *Safety-first countdown for traffic resumption.* |

---

## 🌍 Real-World Impact
Traditional traffic systems are "blind"—they treat a life-saving ambulance the same as a commuter car. 
* **The Problem:** Ambulances lose an average of 45–90 seconds at major urban intersections.
* **The Solution:** IntelliSignal AI provides "Digital Eyes" to city infrastructure, allowing for a **Zero-Wait** environment for emergency responders.

---

## 🚀 Technical Stack
* **Language:** Python 3.10
* **Deep Learning:** YOLOv8 (Ultralytics)
* **Computer Vision:** OpenCV (FFMPEG Backend)
* **Optimization:** PyTorch, CUDA 12.1, Layer Fusion

---

## 🔧 Installation & Setup

1. **Clone the Project:**
   ```bash
   git clone https://github.com/Kishornayak2006/IntelliSignal-with-Emergency-Priority
   cd IntelliSignal-with-Emergency-Priority
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Enable GPU Acceleration:**
   ```bash
   pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

4. **Run the Controller:**
   ```bash
   python controller.py
