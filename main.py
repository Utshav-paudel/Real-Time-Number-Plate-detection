from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv
import csv
import PIL
PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Initialize results dictionary
results = {}

# Initialize the SORT tracker
mot_tracker = Sort()

# Load YOLO models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# IP camera URL
ip_camera_url =  ""  # Example URL from IP Webcam app
cap = cv2.VideoCapture(ip_camera_url)  # You mentioned 'sample.mp4' but I assume you want real-time feed

# Check if the connection is successful
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

vehicles = [2, 3, 5, 7]

# Function to append data to CSV file
def append_to_csv(data, csv_file):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

# Open CSV file and write headers
csv_file = './test.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Car ID', 'Car BBox', 'License Plate BBox', 'License Plate Text', 'BBox Score', 'Text Score'])

# Read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                print("License plate :  ", license_plate_text)
                if license_plate_text is not None:
                    # Save results to dictionary
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

                    # Append results to CSV file
                    append_to_csv([frame_nmr, car_id, [xcar1, ycar1, xcar2, ycar2], [x1, y1, x2, y2], license_plate_text, score, license_plate_text_score], csv_file)

        # Display the frame (optional)
        cv2.imshow('IP Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
