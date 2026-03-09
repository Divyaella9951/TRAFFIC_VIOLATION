from ultralytics import YOLO

# Load number plate detection model
model = YOLO("license_plate_detector.pt")

# Run detection on test image
results = model("test.jpeg")

# Print detection results
print(results)