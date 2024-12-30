from ultralytics import YOLO
import cv2
import time
import pandas as pd
import os
import torch

# Check CUDA
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
print(torch.backends.cudnn.enabled)

# Initialize CSV logging
csv_file = "detection_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w") as f:
        f.write("Frame,Model,Class,Confidence,X1,Y1,X2,Y2\n")  # Write header

# Start webcam
cap = cv2.VideoCapture(0)
original_width = int(cap.get(3))  # Original frame width
original_height = int(cap.get(4))  # Original frame height

cap.set(3, 640)
cap.set(4, 640)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO models
model1 = YOLO("yolo-Weights/yolov10x.pt").to(device)
model2 = YOLO("yolo-Weights/yolo11x.pt").to(device)

classNames = {**model1.model.names, **model2.model.names}
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

frame_count = 0
log_data = []
fps_list = []

def process_results(results, source, scale_x, scale_y):
    detections = []
    for box in results[0].boxes:
        # Extract and scale coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        confidence = round(float(box.conf[0]), 2)
        if confidence < 0.5:
            continue
        cls = int(box.cls[0])
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'confidence': confidence,
            'class_id': cls,
            'source': source
        })
    return detections

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    resized_frame = cv2.resize(frame, (480, 480))  # Resize for faster inference

    # Scaling factors for bounding box adjustment
    scale_x = original_width / 480
    scale_y = original_height / 480

    # Perform YOLO predictions
    results1 = model1.predict(resized_frame, verbose=False)
    results2 = model2.predict(resized_frame, verbose=False)

    detections1 = process_results(results1, source="YOLOv10", scale_x=scale_x, scale_y=scale_y)
    detections2 = process_results(results2, source="YOLOv11", scale_x=scale_x, scale_y=scale_y)

    merged_detections = detections1 + detections2

    # Annotate the frame
    for det in merged_detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{classNames[det['class_id']]} {det['confidence']:.2f}"
        color = colors[det['class_id'] % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add to log data
        log_data.append(f"{frame_count},{det['source']},{classNames[det['class_id']]},{det['confidence']},{x1},{y1},{x2},{y2}")

    # Save logs to CSV every 30 frames
    if frame_count % 30 == 0 and log_data:
        with open(csv_file, "a") as f:
            f.write("\n".join(log_data) + "\n")
        log_data.clear()

    # Calculate FPS
    fps = 1 / (time.time() - start_time)
    fps_list.append(fps)
    avg_fps = sum(fps_list[-30:]) / min(len(fps_list), 30)  # Average FPS over the last 30 frames
    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLO Detection", frame)
    frame_count += 1

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final save to CSV (if any unsaved logs remain)
if log_data:
    with open(csv_file, "a") as f:
        f.write("\n".join(log_data) + "\n")

cap.release()
cv2.destroyAllWindows()
