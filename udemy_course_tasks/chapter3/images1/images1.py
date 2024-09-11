import cv2 as cv  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting images
from IPython.display import Image  # Import Image from IPython for displaying images in Jupyter

# Load the image in grayscale mode
image = cv.imread('D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\chapter4\\bw.png', cv.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if image is None:
    raise FileNotFoundError("The image file 'bw.png' was not found.")

# Convert the image matrix to a string format
matrix_string = np.array2string(image, separator=',', threshold=np.inf)

# Write the matrix to a text file, clearing its content first
with open('D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter4\\images1\\matrix1.txt', 'w') as file:
    file.write(matrix_string)

print("Matrix has been written to 'matrix1.txt'.")

# Convert the grayscale image to RGB format.
# OpenCV loads images in BGR format by default, so this conversion is necessary for correct display using Matplotlib.
img_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)  # Use cv.COLOR_GRAY2RGB for grayscale to RGB conversion

# Display the image using Matplotlib in RGB format
plt.imshow(img_rgb)

# Hide the axes for a cleaner display of the image
plt.axis('off')

# Show the image in the output window or notebook (if using Jupyter)
plt.show()

# Save the RGB image to a file named 'saved_ph1.png'
# Note: The saved image will be in RGB format even though OpenCV saves images in BGR by default.
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter4\\images1\\saved_bw1.png", img_rgb)
