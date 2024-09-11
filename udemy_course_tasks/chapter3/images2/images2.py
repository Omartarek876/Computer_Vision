import cv2 as cv  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting images
from IPython.display import Image  # Import Image from IPython for displaying images in Jupyter

# Load the image in grayscale mode
# 'bw.png' is the image file name
# cv.IMREAD_GRAYSCALE flag indicates that the image will be loaded in grayscale
image = cv.imread('D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\chapter4\\bw.png', cv.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
# If the image is not found or couldn't be loaded, raise an error
if image is None:
    raise FileNotFoundError("The image file 'bw.png' was not found.")

# Create a copy of the original image for modification
# This prevents the original image from being altered
image_copy = image.copy()

# Modify the image by setting a 10x10 square of pixels to 255 (white)
# The loop iterates over a range of pixel indices and changes their values
for i in range(2, 12):  # Loop through rows from 2 to 11
    for j in range(2, 12):  # Loop through columns from 2 to 11
        image_copy[i, j] = 255  # Set the pixel value to 255 (white)

# Convert the modified image matrix to a string format
# The matrix_string will represent the image's pixel values as a string
matrix_string = np.array2string(image_copy, separator=',', threshold=np.inf)

# Write the matrix to a text file, clearing its content first
# 'matrix.txt' will contain the pixel values of the modified image
with open('D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter4\\images2\\matrix2.txt', 'w') as file:
    file.write(matrix_string)

# Print a message to confirm that the matrix has been written to the file
print("Matrix has been written to 'matrix2.txt'.")

# Convert the grayscale image to RGB format
# This is necessary because Matplotlib expects images in RGB format for correct color display
img_rgb = cv.cvtColor(image_copy, cv.COLOR_GRAY2RGB)  # Use cv.COLOR_GRAY2RGB for grayscale to RGB conversion

# Display the image using Matplotlib in RGB format
plt.imshow(img_rgb)

# Hide the axes for a cleaner display of the image
plt.axis('off')

# Show the image in the output window or notebook (if using Jupyter)
plt.show()

# Save the RGB image to a file named 'saved_ph1.png'
# Note: The saved image will be in RGB format even though OpenCV saves images in BGR by default
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter4\\images2\\saved_bw2.png", img_rgb)
