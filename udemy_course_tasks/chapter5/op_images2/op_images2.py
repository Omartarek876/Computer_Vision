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
# OpenCV loads images in BGR format, but this conversion is necessary for correct display in Matplotlib
img_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)


# Create two matrices to adjust contrast: one for lower contrast and one for higher contrast
# The matrix values represent scaling factors for pixel intensities
# Multiplying by 0.8 lowers contrast (makes it darker)
matrix1 = np.ones(img_rgb.shape, dtype=img_rgb.dtype) * 0.8

# Multiplying by 1.2 increases contrast (makes it brighter)
matrix2 = np.ones(img_rgb.shape, dtype=img_rgb.dtype) * 1.2

# Apply the contrast changes by multiplying the original image by the respective matrices
# Convert the image to float before applying multiplication, and then convert it back to uint8
img_darker = np.uint8(cv.multiply(np.float64(img_rgb), matrix1))  # Lower contrast
img_brighter = np.uint8(cv.multiply(np.float64(img_rgb), matrix2))  # Higher contrast


# Set up the figure for displaying images with a size of 12x6 inches
plt.figure(figsize=[12, 6])

# Display the higher contrast image in the first subplot
plt.subplot(141)
plt.imshow(img_brighter)
plt.title("higher contrast")

# Display the lower contrast image in the second subplot
plt.subplot(142)
plt.imshow(img_darker)
plt.title("lower contrast")

# Display the original image in the third subplot
plt.subplot(143)
plt.imshow(img_rgb)
plt.title("original")

# Show all subplots
plt.show()


# Save the higher contrast image to a specified path
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter5\\op_images2\\saved_brighter.png", img_brighter)

# Save the lower contrast image to a specified path
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter5\\op_images2\\saved_darker.png", img_darker)

# Save the original image to a specified path
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter5\\op_images2\\saved_original.png", img_rgb)
