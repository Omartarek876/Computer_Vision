import cv2 as cv  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting images
from IPython.display import Image  # Import Image from IPython for displaying images in Jupyter

src = "car.mp4"
cap = cv.VideoCapture(src)

if not cap.isOpened(): 
    print("Error opening video stream")

ret, frame = cap.read()

if ret:
    plt.imshow(frame[..., ::-1])
    plt.show()
else:
    print("Failed to capture frame")
