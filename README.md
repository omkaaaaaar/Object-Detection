🧠 Object Detection using YOLOv8 and OpenCV (Python)

This project demonstrates how to implement **real-time object detection** using **YOLOv8** (You Only Look Once) and **OpenCV** in Python. It uses a pretrained YOLOv8 model to detect and annotate objects in video streams from a webcam or video file.

## 🚀 Features

- 🎯 Real-time object detection using **YOLOv8n**
- 🧠 Integrates **Ultralytics YOLOv8** with OpenCV for live visualization
- 📦 Runs on CPU or GPU (if available)
- 🔍 Displays bounding boxes, class labels, and confidence scores
- 📸 Works with webcam or video/image files

## 📂 Project Structure

```
📁 yolo-object-detection/
│
├── object_detection.py    # Main script to run YOLOv8 detection with OpenCV
├── yolov8n.pt             # Pretrained YOLOv8n model (lightweight & fast)
├── README.md              # Project overview and usage instructions
```

## 🧰 Requirements

Install the required Python packages:

```bash
pip install ultralytics opencv-python
```

> ✅ Ensure your Python version is 3.7 or higher.

## 🧪 How to Run

Run the object detection script using your webcam:

```bash
python object_detection.py
```

If you want to use a video file instead, modify `cv2.VideoCapture(0)` to `cv2.VideoCapture("your_video.mp4")` in the script.

(Use Video - Preferred)

## 📷 Example Output

- Objects detected with bounding boxes and confidence percentages
- Real-time performance even on low-end systems with `yolov8n.pt`

## ⚙️ How It Works

1. **Load Model**: Loads the YOLOv8n pretrained weights using `ultralytics` library.
2. **Capture Video**: Uses OpenCV to access webcam or video input.
3. **Predict Objects**: Runs the frame through the YOLOv8 model.
4. **Draw Results**: Annotates the frame with bounding boxes, class names, and scores.
5. **Display**: Shows the live detection feed in a resizable OpenCV window.

## 🧠 Model Info

- Model: **YOLOv8n** (Nano version)
- Source: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Format: PyTorch `.pt` file

## 🖼️ Demo

https://www.linkedin.com/feed/update/urn:li:activity:7327599840098451456/

## 📌 Notes

- For best results, ensure good lighting conditions.
- You can try other YOLOv8 variants (e.g., yolov8s, yolov8m) by changing the model path.
- Supports GPU acceleration with CUDA if available.

## 🧑‍💻 Author

Omkar Patkar
 - Artificial Intelligence & Data Science Engineering student at Konkan Gyanpeeth College of Engineering

## 📜 License

This project is licensed under the MIT License. See `LICENSE` file for details.
