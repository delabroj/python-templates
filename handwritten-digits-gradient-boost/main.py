# Import models and utility functions
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn import metrics
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist


# Setting SEED for reproducibility
SEED = 23

# Importing the dataset
# X, y = load_digits(return_X_y=True)

# Splitting dataset
# train_X, test_X, train_y, test_y = train_test_split(
#     X, y, test_size=0.25, random_state=SEED
# )

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)


# Creating Augmented Dataset


# Method to shift the image by given dimension
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


X_train_augmented = [image for image in train_X]
y_train_augmented = [image for image in train_y]

print("creating augmented data source")
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(train_X, train_y):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)


# Shuffle the dataset
shuffle_idx = np.random.permutation(len(X_train_augmented))
train_X = np.array(X_train_augmented)[shuffle_idx]
train_y = np.array(y_train_augmented)[shuffle_idx]

# Instantiate Gradient Boosting Classifier
gbc = HistGradientBoostingClassifier(
    max_iter=2000, learning_rate=0.05, random_state=100, max_bins=10, verbose=1
)
# Fit to training set
result = gbc.fit(train_X, train_y)
print(result.n_iter_)

# Predict on test set
pred_y = gbc.predict(test_X)
print(pred_y)
print(test_y)

# accuracy
acc = accuracy_score(test_y, pred_y)
print("Gradient Boosting Classifier accuracy is : {:.3f}".format(acc))

disp = metrics.ConfusionMatrixDisplay.from_predictions(test_y, pred_y)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
