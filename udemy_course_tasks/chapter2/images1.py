import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
from IPython.display import Image 

# Display the image using IPython for a preview in notebook environments (does not perform any actual image processing).
Image("road.jpg")

# Load the image using OpenCV in color mode. The flag '1' indicates loading as a color image in BGR format.
img = cv.imread("road.jpg", 1)

# Print the shape of the image, which returns a tuple (height, width, number of channels).
# Example output: (height, width, 3), where 3 represents the color channels (BGR).
print(img.shape)

# Print the data type of the image. Typically, it is 'uint8', meaning each pixel value is represented using 8 bits.
print(img.dtype)

# Convert the image from BGR (which is the default format for OpenCV) to RGB format.
# This conversion is required to display the image correctly using matplotlib, which expects RGB format.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Display the image using matplotlib in RGB format.
plt.imshow(img_rgb)

# Hide the axes for a cleaner display of the image.
plt.axis('off')

# Show the image on the output window or in the notebook (if you're using Jupyter).
plt.show()

# Save the RGB image to a file named 'saved_ph1.png'.
# Although OpenCV saves images in BGR by default, we are saving the RGB image here since we converted it.
cv.imwrite("saved_ph1.png", img_rgb)
