# YOLO Object Detection on a Folder of Images

## Author: Omar Tarek Ibrahim

## Project Description:
This project implements object detection using the YOLOv3 (You Only Look Once) deep learning model. The script processes a folder of images, detects objects, and saves the processed images with bounding boxes and class labels for the detected objects. The model uses the pre-trained COCO dataset, which can detect 80 different object classes.

## Features:
- Processes all images in the `input_images` folder.
- Uses YOLOv3 for real-time object detection.
- Draws bounding boxes around detected objects and labels them with class names.
- Saves processed images with bounding boxes to the `output_images` folder.

## Requirements:
The project requires Python 3 and the following libraries:
- `opencv-python`
- `numpy`

You can install the required dependencies using the following command:
```bash
pip install opencv-python numpy
