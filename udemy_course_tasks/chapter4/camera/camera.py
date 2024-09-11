import cv2 as cv  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting images
from IPython.display import Image  # Import Image from IPython for displaying images in Jupyter


alive =True 
win_name = "accessing camera"

cv.namedWindow(win_name , cv.WINDOW_NORMAL)
src = cv.VideoCapture(0)

while alive : 
    ret , frame = src.read()

    if not ret : 
        break 

    cv.imshow(win_name , frame)

    if cv.waitKey(1) == ord('q') : 
        alive = False 