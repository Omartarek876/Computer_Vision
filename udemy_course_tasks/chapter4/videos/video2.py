import cv2 as cv  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting images
from IPython.display import Image  # Import Image from IPython for displaying images in Jupyter

src = "car.mp4"
cap = cv.VideoCapture(src)

if not cap.isOpened(): 
    print("Error opening video stream")

frame_width = int(cap.get(3))  # Get the width of the frames
frame_height = int(cap.get(4))  # Get the height of the frames

# Correctly initialize the VideoWriter with the appropriate fourcc codec
out_mp4 = cv.VideoWriter("out.mp4", cv.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))

# Example loop to read from the video and write to the output file
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames are available

    out_mp4.write(frame)  # Write the frame to the output video

cap.release()  # Release the video capture object
out_mp4.release()  # Release the video writer object
