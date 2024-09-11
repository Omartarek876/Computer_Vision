import cv2 as cv  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting images
from IPython.display import Image  # Import Image from IPython for displaying images in Jupyter


# Load the image in grayscale mode
image = cv.imread('moun.png')

# Check if the image was loaded correctly
if image is None:
    raise FileNotFoundError("The image file 'bw.png' was not found.")


# Convert the grayscale image to RGB format for Matplotlib
# OpenCV loads images in BGR format by default, but this conversion is necessary for correct display using Matplotlib
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

feature_params = dict( maxCorners = 25,
                       qualityLevel = 0.2,
                       minDistance = 15,
                       blockSize = 9 )

corners = cv.goodFeaturesToTrack(gray, **feature_params)
corners = np.intp(corners)

if corners is not None:
    for x, y in np.float32(corners).reshape(-1, 2):
        cv.circle(image, (int(x), int(y)), 2, (0, 255, 0), 1)

img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()


# Save the blurred image to a specified path
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter5\\op_images4\\saved_cornered.png", image)
