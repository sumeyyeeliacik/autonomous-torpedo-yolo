# Autonomous Torpedo System with YOLO

This project is part of an Autonomous Underwater Vehicle (AUV) developed by our student robotics team. It implements an AI-based torpedo system using real-time object detection with a YOLOv5 model in ONNX format. Once a circular target is detected and aligned, the system executes a simulated torpedo launch based on decision logic. 

---

## Features
- Real-time object detection using YOLOv5 (ONNX).
- Python-based control logic using OpenCV and DNN module.
- Underwater navigation and target tracking.
- Conditional torpedo activation after alignment and size verification.
- Search, align, and fire behavior implemented with logic layers.

---

## File Structure

├── torpedo2.py # Main control and detection script

## Model

Due to GitHub file size restrictions, the trained YOLOv5 model (ONNX format) is shared via Google Drive:

https://drive.google.com/drive/folders/1ZKGtFRwLtFkJ7JWA4WTt1OuVZa42nCes?usp=sharing

---

## 📷 Sample Detection Frame

This frame shows the moment the torpedo system detects a circular target using the trained YOLO model:

![frame_0286](https://github.com/user-attachments/assets/b7cb165b-6e17-4e4b-9e46-000048e897f8)



---

##  Technologies Used
- Python 3.8
- OpenCV (cv2)
- PyTorch (for model training)
- YOLOv5 → converted to ONNX
- Autonomous vehicle control logic

---

## Author

**Sümeyye Eliaçık**  
Second-year Computer Engineering student  
Passionate about AI, robotics, and real-world problem solving.

---
