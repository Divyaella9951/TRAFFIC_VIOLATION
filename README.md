# TRAFFIC_VIOLATION
AI-based Smart Traffic Monitoring System using YOLOv8, OpenCV, PyTorch, and Flask to detect helmet violations, seatbelt usage, driver distractions, and automatically extract license plate numbers with OCR.



# Smart Traffic Monitoring System

An AI-powered **Smart Traffic Monitoring System** that automatically detects traffic violations using computer vision and deep learning.
The system analyzes live camera feed to detect **helmet usage, seatbelt usage, driver distractions, and vehicle license plates**, and logs violations with fine details.

This project demonstrates how **AI can be used to improve road safety and automate traffic law enforcement**.

---

# Features

 Helmet detection using **YOLOv8**
 Seatbelt detection using **YOLOv8**
 Driver distraction detection using **ResNet18 (PyTorch)**
 Automatic **license plate detection**
 License plate recognition using **EasyOCR**
 **Live camera monitoring dashboard (Flask)**
 Automatic **violation detection and fine calculation**
 **CSV violation record logging**
 Real-time **traffic monitoring interface**

---

# Technologies Used

* **Python**
* **OpenCV**
* **YOLOv8 (Ultralytics)**
* **PyTorch**
* **EasyOCR**
* **Flask**
* **HTML / CSS / JavaScript**

---

# System Workflow

Camera Feed
⬇
Helmet Detection (YOLO)
⬇
Seatbelt Detection (YOLO)
⬇
Driver Distraction Detection (ResNet18)
⬇
License Plate Detection
⬇
OCR Plate Recognition
⬇
Violation Detection
⬇
Fine Calculation
⬇
Dashboard Display + CSV Logging

---

# Violation Rules

| Violation          | Fine  |
| ------------------ | ----- |
| No Helmet          | ₹1000 |
| No Seatbelt        | ₹500  |
| Driver Distraction | ₹1500 |

---

# Project Structure

```
Smart-Traffic-Monitoring-System
│
├── app.py
├── master_detection.py
├── license_plate_detector.pt
├── violations_record.csv
│
├── templates
│   └── index.html
│
├── static
│   ├── style.css
│   └── script.js
│
├── violations
│   └── violation_images
│
├── plates
│   └── detected_plates
│
└── datasets
    └── trained_models
```

---

#  Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/smart-traffic-monitoring-system.git
cd smart-traffic-monitoring-system
```

Install dependencies:

```bash
pip install ultralytics
pip install opencv-python
pip install torch torchvision
pip install flask
pip install easyocr
pip install pillow
```

---

#  Run the Project

Start the application:

```bash
python app.py
```

Open the dashboard in your browser:

```
http://127.0.0.1:5000
```

---

#  Dashboard

The dashboard displays:

* Date & Time
* Camera Location
* Helmet Status
* Seatbelt Status
* Driver Activity
* Vehicle Number
* Violation Type
* Fine Amount

---

#  Violation Records

All violations are stored in:

```
violations_record.csv
```

Example:

```
Date & Time,Camera Location,Helmet Status,Seatbelt Status,Driver Activity,Vehicle Number,Violation Type,Fine Amount
08-03-2026 21:15:22,Hyderabad Junction,Not Detected,Detected,Talking on Phone,TS09AB1234,No Helmet + Driver Distraction,2500
```

---

#  Future Improvements

* Multi-camera traffic monitoring
* Cloud database integration
* Automatic fine payment system
* Vehicle tracking
* Real-time traffic analytics
* Integration with smart city infrastructure






