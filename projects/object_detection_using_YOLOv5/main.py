"""
Author: Omar Tarek Ibrahim 
E-mail: OmarrTarek74@gmail.com
File Name : main.py
Last Updated Date: 11/9/2024 
Project: YOLO Object Detection on a Folder of Images
Description: This project loads the YOLOv3 object detection model and processes 
             a folder of images to detect objects, draw bounding boxes, and 
             save the output images with bounding boxes and labels in a separate folder.
"""

import cv2 as cv
import numpy as np
import os

# Load YOLO model with the pre-trained weights and configuration
yolo = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the class names from coco.names file (used for labeling detected objects)
classes = []
with open("coco.names", 'r') as file:
    classes = [line.strip() for line in file.readlines()]

# Get the names of the layers in the YOLO network
layer_names = yolo.getLayerNames()

# Get the indices of the output layers for YOLO (unconnected layers are the final detection layers)
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers().flatten()]

# Define colors for bounding boxes: Red for box, Green for text labels
colorRed = (0, 0, 255)
colorGreen = (0, 255, 0)

# Set the input and output folder paths
input_folder = "input_images"  # Folder containing the images to process
output_folder = "output_images"  # Folder to save the processed images

# Create the output folder if it doesn't already exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each file in the input folder
for image_name in os.listdir(input_folder):
    # Only process image files (with specific extensions)
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Construct the full path to the image
        img_path = os.path.join(input_folder, image_name)

        # Read the image
        img = cv.imread(img_path)
        if img is None:
            print(f"Could not read image {image_name}")
            continue  # Skip if the image could not be loaded

        # Get the image dimensions (height, width, channels)
        height, width, channels = img.shape

        # Preprocess the image to create a blob for YOLO input
        blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Set the YOLO model input
        yolo.setInput(blob)

        # Run the forward pass to get predictions from the output layers
        outputs = yolo.forward(output_layers)

        # Lists to store information about detected objects
        class_ids = []  # Detected class IDs (object types)
        confidences = []  # Confidence scores of the detections
        boxes = []  # Bounding box coordinates

        # Process each output from YOLO
        for output in outputs:
            for detection in output:
                # Extract the confidence scores for each object class
                scores = detection[5:]
                class_id = np.argmax(scores)  # Get the class with the highest confidence
                confidence = scores[class_id]

                # Only consider objects with confidence above a threshold (e.g., 0.5)
                if confidence > 0.5:
                    # Get the bounding box coordinates relative to the image size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Convert center coordinates to top-left corner coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Store the bounding box, confidence, and class ID
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maxima Suppression (NMS) to remove overlapping boxes
        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Flatten the indexes if needed (to avoid potential errors)
        indexes = indexes.flatten() if len(indexes) > 0 else []

        # Draw the bounding boxes and labels on the image
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])

                # Draw the bounding box in red
                cv.rectangle(img, (x, y), (x + w, y + h), colorRed, 2)

                # Draw the label (class name) in green above the bounding box
                cv.putText(img, label, (x, y - 10), cv.FONT_HERSHEY_PLAIN, 1, colorGreen, 2)

        # Save the output image with bounding boxes to the output folder
        output_path = os.path.join(output_folder, image_name)
        cv.imwrite(output_path, img)

        print(f"Processed {image_name} and saved to {output_path}")

# Print a message when all images have been processed
print("Processing complete.")

