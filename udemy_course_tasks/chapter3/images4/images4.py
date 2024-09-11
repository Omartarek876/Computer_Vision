import cv2 as cv  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting images
from IPython.display import Image  # Import Image from IPython for displaying images in Jupyter

# Load the image in grayscale mode
# Provide the full path to the image file. Here, 'cv.IMREAD_GRAYSCALE' loads the image as grayscale.
image = cv.imread('D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter4\\road.png', cv.IMREAD_COLOR)

# Check if the image was loaded correctly
# If the image is not found or couldn't be loaded, raise a FileNotFoundError.
if image is None:
    raise FileNotFoundError("The image file 'bw.png' was not found.")

# Print the shape of the original image to see its dimensions (rows, columns)
print(image.shape)

# Resize the image with scaling factors fx and fy
# 'fx' and 'fy' are the scaling factors along the x and y axes respectively.
# This enlarges the image by 4 times along the x-axis and 2 times along the y-axis.
cropped_img = image [200:700 , 200:800]

# Print the shape of the resized image to verify the new dimensions
print(cropped_img.shape)

# Convert the resized grayscale image to RGB format
# This is necessary because Matplotlib expects images in RGB format for correct color display.
img_rgb = cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB)  # Use cv.COLOR_GRAY2RGB for grayscale to RGB conversion

# Display the image using Matplotlib in RGB format
plt.imshow(img_rgb)

# # Hide the axes for a cleaner display of the image
# plt.axis('off')

# Show the image in the output window or notebook (if using Jupyter)
plt.show()

# Uncomment the following lines to save the RGB image to a file
# The image will be saved in RGB format even though OpenCV saves images in BGR by default.
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter4\\images4\\saved_bw4.png", img_rgb)
