import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("birds.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(1)
plt.title("original")
plt.imshow(gray, cmap="gray")

# Blur to remove noise
blur = cv2.GaussianBlur(gray, (11, 11), 0)
plt.figure(2)
plt.title("blured")
plt.imshow(blur, cmap="gray")

# Threshold blured image to sharpen edges
_, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.figure(3)
plt.title("threshold")
plt.imshow(threshold, cmap="gray")

# Detect edges using canny algorithm
canny = cv2.Canny(threshold, 10, 30, 3)
plt.figure(4)
plt.title("canny edges")
plt.imshow(canny, cmap="gray")

# # Expand canny selections so segments are combined
# dilated = cv2.dilate(canny, (10, 10), iterations=1)
# plt.figure(5)
# plt.title("dilated")
# plt.imshow(blur, cmap="gray")

# Find contours
(cnt, hierarchy) = cv2.findContours(
    canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)

# Draw contours onto original image
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, 0, (255, 255, 0), 2)
cv2.drawContours(rgb, cnt, 1, (0, 255, 0), 2)
cv2.drawContours(rgb, cnt, 2, (0, 255, 255), 2)
cv2.drawContours(rgb, cnt, 3, (255, 0, 0), 2)
cv2.drawContours(rgb, cnt, 4, (255, 0, 255), 2)
cv2.drawContours(rgb, cnt, 5, (0, 100, 100), 2)
cv2.drawContours(rgb, cnt, 6, (255, 100, 0), 2)
cv2.drawContours(rgb, cnt, 7, (100, 0, 255), 2)
plt.figure(6)
plt.title("contours")
plt.imshow(rgb)


print("birds in the image: ", len(cnt))

plt.show()
