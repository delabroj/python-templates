import numpy as np
import cv2


# Read image
image = cv2.imread("digits.png")

# Convert to gray scale
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Divide the image into 5000 small images of size 20x20
divisions = list(np.hsplit(i, 100) for i in np.vsplit(gray_img, 50))

# Convert into Numpy array of size (50,100,20,20)
NP_array = np.array(divisions)

# Split into train and test data
train_data = NP_array[:, :50].reshape(-1, 400).astype(np.float32)
test_data = NP_array[:, 50:100].reshape(-1, 400).astype(np.float32)

# Add labels
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = np.repeat(k, 250)[:, np.newaxis]

# Create kNN classifier
knn = cv2.ml.KNearest_create()

# Train classifier
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Find nearest neighbors to get test_data predictions
ret, output, neighbours, distance = knn.findNearest(test_data, k=3)

# Check accuracy
matched = output == test_labels
correct_OP = np.count_nonzero(matched)

accuracy = (correct_OP * 100.0) / (output.size)

print(accuracy)
