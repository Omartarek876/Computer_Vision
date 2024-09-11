import cv2 as cv  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting images
from IPython.display import Image  # Import Image from IPython for displaying images in Jupyter


# Load the image in grayscale mode
image = cv.imread('moun.png', cv.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if image is None:
    raise FileNotFoundError("The image file 'bw.png' was not found.")

# Convert the grayscale image to RGB format for Matplotlib
# OpenCV loads images in BGR format by default, but this conversion is necessary for correct display using Matplotlib
img_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)


# Apply a blur effect to the image using a 25x25 kernel
# The larger the kernel size, the more blurred the image becomes
blur = cv.blur(img_rgb, (25, 25))


# Set up the figure for displaying images with a size of 12x6 inches
plt.figure(figsize=[12, 6])

# Display the blurred image in the first subplot
plt.subplot(141)
plt.imshow(blur)
plt.title("blurred")

# Display the original image in the third subplot
plt.subplot(143)
plt.imshow(img_rgb)
plt.title("original")

# Show all subplots
plt.show()


# Save the blurred image to a specified path
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter5\\op_images3\\saved_blurred.png", blur)

# Save the original image to a specified path
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter5\\op_images2\\saved_original.png", img_rgb)
