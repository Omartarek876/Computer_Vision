# Object Detection Using OpenCV DNN and Webcam

- This project implements real-time object detection using a pre-trained object detection model (SSD MobileNet V3) and a webcam feed ,It captures video from a mobile camera via an IP stream, detects objects, and displays the results in a window.
- The model is based on the COCO dataset and is capable of detecting and labeling objects with bounding boxes.

## Features
- **Real-time object detection** using OpenCV’s DNN module.
- **SSD MobileNet V3** pre-trained on the COCO dataset for detecting common objects.
- **Webcam integration** for capturing and displaying object detection in real-time.
- **Image capture**: Press the `c` key to save a snapshot with bounding boxes.
- **Exit**: Press the `q` key to quit the application.

## Requirements
To run the project, you need the following dependencies:
- Python 3.x
- OpenCV 4.x (`pip install opencv-python`)
- Pre-trained SSD MobileNet V3 model files:
  - `frozen_inference_graph.pb` (Model weights)
  - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` (Model configuration)
- COCO dataset class names file: `coco.names`
