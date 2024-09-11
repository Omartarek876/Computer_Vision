import cv2 as cv  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting images
from IPython.display import Image  # Import Image from IPython for displaying images in Jupyter


# Load the image in grayscale mode
image = cv.imread('moun.png', cv.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if image is None:
    raise FileNotFoundError("The image file 'bw.png' was not found.")

# Convert the grayscale image to RGB format.
# OpenCV loads images in BGR format by default, so this conversion is necessary for correct display using Matplotlib.
img_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)  # Use cv.COLOR_GRAY2RGB for grayscale to RGB conversion


# Create a matrix of ones with the same shape as the image, to adjust the brightness
# The dtype of the matrix matches that of the image, and each pixel is incremented by 10
matrix = np.ones(img_rgb.shape, dtype=img_rgb.dtype) * 10

# Add the matrix to make the image brighter
img_brighter = cv.add(img_rgb, matrix)

# Subtract the matrix to make the image darker
img_darker = cv.subtract(img_rgb, matrix)


# Set up the figure for plotting images with a size of 12x6 inches
plt.figure(figsize=[12, 6])

# Display the brighter image in the first subplot
plt.subplot(141)
plt.imshow(img_brighter)
plt.title("brighter")

# Display the darker image in the second subplot
plt.subplot(142)
plt.imshow(img_darker)
plt.title("darker")

# Display the original image in the third subplot
plt.subplot(143)
plt.imshow(img_rgb)
plt.title("original")

# Show all subplots
plt.show()


# Save the brighter image to a specified path
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter5\\op_images1\\saved_brighter.png", img_brighter)

# Save the darker image to a specified path
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter5\\op_images1\\saved_darker.png", img_darker)

# Save the original image to a specified path
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter5\\op_images1\\saved_original.png", img_rgb)
