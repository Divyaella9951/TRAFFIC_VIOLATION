from ultralytics import YOLO
import torch
import cv2
import os
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import easyocr
import csv


# ================= CREATE FOLDERS =================
os.makedirs("violations", exist_ok=True)
os.makedirs("plates", exist_ok=True)

# ================= LOAD MODELS =================
helmet_model = YOLO("datasets/archivee/helmet_dataset/runs/detect/train8/weights/best.pt")
seatbelt_model = YOLO("datasets/seatbelt_dataset/runs/detect/train/weights/best.pt")
plate_model = YOLO("license_plate_detector.pt")

# ================= DRIVER DISTRACTION MODEL =================
distraction_model = models.resnet18()
distraction_model.fc = torch.nn.Linear(distraction_model.fc.in_features, 10)

distraction_model.load_state_dict(
    torch.load(
        "datasets/archive/driver_distraction_model.pth",
        map_location=torch.device('cpu')
    )
)

distraction_model.eval()

print("All models loaded successfully")

# ================= DRIVER LABELS =================
driver_labels = [
    "Safe Driving",
    "Talking on Phone",
    "Texting",
    "Drinking",
    "Reaching Behind",
    "Hair & Makeup",
    "Adjusting Radio",
    "Talking to Passenger",
    "Operating GPS",
    "Looking Back"
]

# ================= IMAGE TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ================= OCR =================
reader = easyocr.Reader(['en'])

# ================= CAMERA =================
def start_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not working")
    exit()

print("Camera started successfully")

# ================= FINE RULES =================
fine_rules = {
    "No Helmet": 1000,
    "No Seatbelt": 500,
    "Driver Distraction": 1500
}

# ================= MAIN LOOP =================
def run_detection():

    ret, frame = cap.read()

    if not ret:
        return {"error":"Camera not working"}

    violations = []

    

    # ================= HELMET DETECTION =================
    helmet_results = helmet_model(frame, conf=0.6)
    helmet_detected = False

    for r in helmet_results:
        if len(r.boxes) > 0:
            helmet_detected = True

    if not helmet_detected:
        violations.append("No Helmet")

    # ================= SEATBELT DETECTION =================
    seatbelt_results = seatbelt_model(frame, conf=0.6)
    seatbelt_detected = False

    for r in seatbelt_results:
        if len(r.boxes) > 0:
            seatbelt_detected = True

    if not seatbelt_detected:
        violations.append("No Seatbelt")

    # ================= DRIVER DISTRACTION =================
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0)

    outputs = distraction_model(img)
    _, predicted = torch.max(outputs, 1)

    activity = driver_labels[predicted.item()]

    if activity != "Safe Driving":
        violations.append("Driver Distraction")

    # ================= DISPLAY STATUS =================
    helmet_status = "Detected" if helmet_detected else "Not Detected"
    seatbelt_status = "Detected" if seatbelt_detected else "Not Detected"

    cv2.putText(frame, f"Helmet: {helmet_status}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Seatbelt: {seatbelt_status}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Driver Activity: {activity}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # ================= IF VIOLATION =================
    if len(violations) > 0:

        cv2.putText(frame,
                    "TRAFFIC VIOLATION",
                    (20,170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"violations/violation_{timestamp}.jpg"

        cv2.imwrite(image_path, frame)

        # ================= LICENSE PLATE DETECTION =================
        img = cv2.imread(image_path)
        results = plate_model(img)

        plate = None
        plate_path = "plates/detected_plate.jpg"

        for r in results:
            boxes = r.boxes.xyxy
            for box in boxes:
                x1,y1,x2,y2 = map(int,box)
                plate = img[y1:y2, x1:x2]
                cv2.imwrite(plate_path, plate)

        # ================= OCR =================
        if plate is not None:

            plate_img = cv2.imread(plate_path)
            plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

            result = reader.readtext(plate_gray)

            if len(result) > 0:
                plate_number = result[0][1]
            else:
                plate_number = "Not Detected"

        else:
            plate_number = "Plate Not Found"

        # ================= FINE CALCULATION =================
        total_fine = 0

        for v in violations:
            total_fine += fine_rules[v]

        violation_text = " + ".join(violations)
        # calculate current date and time
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        # calculate location (for demo we will use a fixed location)
        camera_location = "Hyderabad Junction"
        data = {
        "time": current_time,
        "location": camera_location,
        "helmet": helmet_status,
        "seatbelt": seatbelt_status,
        "activity": activity,
        "vehicle": plate_number,
        "violation": violation_text,
        "fine": total_fine
        
        }
        

       
        
        # ================= PROFESSIONAL OUTPUT =================
        print("\nSMART TRAFFIC MONITORING SYSTEM")
        print("--------------------------------\n")

        print(f"Date & Time        : {current_time}")
        print(f"Camera Location    : {camera_location}")
        print(f"Helmet Status      : {helmet_status}")
        print(f"Seatbelt Status    : {seatbelt_status}")
        print(f"Activity           : {activity}")
        print(f"Vehicle Number     : {plate_number}")
        print(f"Violation Type     : {violation_text}")
        print(f"Fine Amount        : ₹{total_fine}")

        print("\n⚠ TRAFFIC VIOLATION DETECTED")
        
        print("Please follow traffic safety rules")

        # ================= SAVE CSV =================
        # ================= SAVE CSV =================
        file_exists = os.path.exists("violations_record.csv")

        with open("violations_record.csv","a",newline="") as file:
            writer = csv.writer(file)
            # Write header if file is new
            if not file_exists:
                writer.writerow([
                    "Date & Time",
                    "Camera Location",
                    "Helmet Status",
                    "Seatbelt Status",
                    "Driver Activity",
                    "Vehicle Number",
                    "Violation Type",
                    "Fine Amount"
                ])
            # Write actual violation data
            writer.writerow([
                current_time,
                camera_location,
                helmet_status,
                seatbelt_status,
                activity,
                plate_number,
                violation_text,
                total_fine
            ])
        return data
    else:

        cv2.putText(frame,
                    "STATUS: SAFE",
                    (20,170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    3)

        print("\nSTATUS: SAFE")
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        camera_location = "Hyderabad Junction"

        data = {
            "time": current_time,
            "location": camera_location,
            "helmet": helmet_status,
            "seatbelt": seatbelt_status,
            "activity": activity,
            "vehicle": "None",
            "violation": "Safe",
            "fine": 0
            }
        return data