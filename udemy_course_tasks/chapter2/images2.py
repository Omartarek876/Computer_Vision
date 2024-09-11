import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
from IPython.display import Image 

# Read the image in BGR format (OpenCV reads images in BGR by default)
img_bgr = cv.imread("road.jpg", cv.IMREAD_COLOR)

# Split the image into its Blue, Green, and Red channels
b, g, r = cv.split(img_bgr)

# Show the individual channels using matplotlib
plt.figure(figsize=[20, 5])  # Set the figure size to 20x5

# Display the Red channel
plt.subplot(141)  # Create a subplot in a 1x4 grid at the 1st position
plt.imshow(r, cmap='gray')   # Display the red channel in grayscale
plt.title("Red Channel")     # Add a title to the subplot

# Display the Green channel
plt.subplot(142)  # Create a subplot in a 1x4 grid at the 2nd position
plt.imshow(g, cmap='gray')   # Display the green channel in grayscale
plt.title("Green Channel")   # Add a title to the subplot

# Display the Blue channel
plt.subplot(143)  # Create a subplot in a 1x4 grid at the 3rd position
plt.imshow(b, cmap='gray')   # Display the blue channel in grayscale
plt.title("Blue Channel")    # Add a title to the subplot

# Merge the individual B, G, R channels back into a single BGR image
img_merged = cv.merge([b, g, r])

# Show the merged image (converted to RGB for correct display in matplotlib)
plt.subplot(144)  # Create a subplot in a 1x4 grid at the 4th position
plt.imshow(img_merged[:, :, ::-1])  # Convert BGR to RGB for display with matplotlib
plt.title("Merged Output")  # Add a title to the subplot

# Show all subplots (individual channels and merged image)
plt.show()

# Save the original BGR image as "saved_ph1.png"
cv.imwrite("saved_ph2.png", img_bgr)  # This saves the image in its original BGR format to a file
