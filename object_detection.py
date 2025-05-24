import cv2
from ultralytics import YOLO

# === CONFIGURATION ===
input_video_path = "input.mp4"   # Ensure this video exists
output_video_path = "output_with_detections.mp4"
model_path = "yolov8n.pt"        # Auto-downloads from ultralytics if not found

# Load YOLOv8 model
model = YOLO(model_path)

# Open input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Class names and target filter
class_names = model.names
relevant_classes = ['person', 'car', 'bicycle', 'motorbike', 'truck', 'bus']

# Bright contrasting colors
person_colors = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (255, 0, 255),   # Magenta
    (255, 255, 0)    # Yellow
]
default_color = (0, 255, 255)    # Cyan for non-person classes

print("Processing video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    person_count = 0  # Track person index

    for box in results.boxes:
        cls_id = int(box.cls)
        label = class_names[cls_id]
        conf = box.conf.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label in relevant_classes:
            if label == "person":
                color = person_colors[person_count % len(person_colors)]
                person_count += 1
            else:
                color = default_color

            # Draw thick rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=4)

            # Draw filled label background
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)

            # Put text on top
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Save frame to output
    out.write(frame)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Saved output video as: {output_video_path}")
