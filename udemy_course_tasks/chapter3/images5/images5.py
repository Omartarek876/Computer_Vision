import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the image in BGR format
image = cv.imread("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter4\\road.png", cv.IMREAD_COLOR)

# Check if the image was loaded correctly
if image is None:
    raise FileNotFoundError("The image file was not found.")
else:
    print("Image loaded successfully:", image.shape)

# Apply flipping operations
flipped_horz = cv.flip(image, 1)
flipped_vert = cv.flip(image, 0)
flipped_both = cv.flip(image, -1)

# Set up the figure with a size of 20x5 inches
plt.figure(figsize=[15, 5])

# Display the horizontally flipped image
plt.subplot(141)
plt.imshow(cv.cvtColor(flipped_horz, cv.COLOR_BGR2RGB))
plt.title("Flipped Horizontally")

# Display the vertically flipped image
plt.subplot(142)
plt.imshow(cv.cvtColor(flipped_vert, cv.COLOR_BGR2RGB))
plt.title("Flipped Vertically")

# Display the image flipped both horizontally and vertically
plt.subplot(143)
plt.imshow(cv.cvtColor(flipped_both, cv.COLOR_BGR2RGB))
plt.title("Flipped Both")

# Display the original image
plt.subplot(144)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Original Image")

# Show all subplots
plt.show()

# Save the horizontally flipped image as "saved_ph5.png"
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter4\\images5\\saved_ph5.1.png", flipped_horz)
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter4\\images5\\saved_ph5.2.png", flipped_vert)
cv.imwrite("D:\\courses\\OpenCV\\mastering_opencv(udemy)\\OpenCV\\chapter4\\images5\\saved_ph5.3.png", flipped_both)
