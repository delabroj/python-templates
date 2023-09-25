import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns

from keras.datasets import mnist
from keras import layers
from sklearn import metrics
from sklearn.metrics import accuracy_score
from scipy.ndimage import shift

(X_train, y_train), (X_test, y_test) = mnist.load_data()


print(X_train.shape)


# Method to shift the image by given dimension
# def shift_image(image, dx, dy):
#     # image = image.reshape((28, 28))
#     shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
#     return shifted_image


# X_train_augmented = [image for image in X_train]
# y_train_augmented = [image for image in y_train]

# print("creating augmented data source")
# for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
#     for image, label in zip(X_train, y_train):
#         X_train_augmented.append(shift_image(image, dx, dy))
#         y_train_augmented.append(label)


# # Shuffle the dataset
# shuffle_idx = np.random.permutation(len(X_train_augmented))
# X_train = np.array(X_train_augmented)[shuffle_idx]
# y_train = np.array(y_train_augmented)[shuffle_idx]

# Check how many examples do we have in our train and test sets
print(
    f"We have {len(X_train)} images in the training set and {len(X_test)} images in the test set."
)

# See the shape of the first sample of our training set
print(X_train[0].shape)

# Plot the first image in our dataset
plt.imshow(X_train[0])
plt.show()

# Plot in grayscale with no axes
plt.figure(figsize=(3, 3))
plt.imshow(X_train[0], cmap="gray")
plt.title(y_train[0])
plt.axis(False)
plt.show()

# Plot a random image
random_image = random.randint(0, len(X_train))

plt.figure(figsize=(3, 3))
plt.imshow(X_train[random_image], cmap="gray")

plt.title(y_train[random_image])
plt.axis(False)
plt.show()

# The Conv2D layer in a convolutional model requires the input to be in shape: [height, width, color_channels] but we only have the height and width dimensions so far.
# Let's reshape our train and test data to have the missing color_channels dimension as well.
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

print(X_train.shape)  # (60000, 28, 28, 1)

# Normalize our train and test images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Change the datatype of our training and test sets to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# We’ll follow the TinyVGG architecture

model = tf.keras.Sequential(
    [
        layers.Conv2D(
            filters=10, kernel_size=3, activation="relu", input_shape=(28, 28, 1)
        ),  # Applies 10 filters to the images
        layers.Conv2D(10, 3, activation="relu"),
        layers.MaxPool2D(),  # Downsizes the images
        layers.Conv2D(10, 3, activation="relu"),
        layers.Conv2D(10, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),  # Make the output one-dimensional
        layers.Dense(10, activation="softmax"),
    ]
)

# Check summary of model
print(model.summary())

# Compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

# Train model or load saved model
# model.fit(X_train, y_train, epochs=10)
model.load_weights("./model_base")

# Check against test set
model.evaluate(X_test, y_test)

# Save model
# model.save("model.bin")

# Predict the value of the digit on the test subset
predicted = model.predict(X_test)

print(y_test.shape)
print(y_test)
# Convert predicted weights into index of highest weight
predicted = np.argmax(predicted, axis=1)
print(predicted.shape)
print(predicted)

# Print classification report
print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

# 4 test samples and show their predicted digit value in the title.
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    ax.imshow(image, cmap="gray", interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# accuracy
acc = accuracy_score(y_test, predicted)
print("Gradient Boosting Classifier accuracy is : {:.3f}".format(acc))

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
