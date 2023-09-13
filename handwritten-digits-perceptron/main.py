from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


digits = datasets.load_digits()

# Display the attributes of the dataset
print(dir(digits))

# Print first image array
print(digits.images[0])


# Plot 16 of the digits from the dataset
def plot_multi(i):
    nplots = 16
    fig = plt.figure(figsize=(15, 15))
    for j in range(nplots):
        plt.subplot(4, 4, j + 1)
        plt.imshow(digits.images[i + j], cmap="binary")
        plt.title(digits.target[i + j])
        plt.axis("off")
    # printing the each digits in the dataset.
    plt.show()


plot_multi(0)


# Flatten the images
y = digits.target
x = digits.images.reshape((len(digits.images), -1))

print(x.shape)
print(x[0])

# Use the first 1000 images for training
x_train = x[:1000]
y_train = y[:1000]

x_test = x[1000:]
y_test = y[1000:]


# Multi-Layer Perceptron
mlp = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="logistic",
    alpha=1e-4,
    solver="sgd",
    tol=1e-6,
    random_state=1,
    learning_rate_init=0.1,
    verbose=True,
    max_iter=1000,
)

mlp.fit(x_train, y_train)

# Plot training progress
fig, axes = plt.subplots(1, 1)
axes.plot(mlp.loss_curve_, "o-")
axes.set_xlabel("number of iteration")
axes.set_ylabel("loss")
plt.show()

# Make predictions from test data
predictions = mlp.predict(x_test)
print(predictions[:50])

print(y_test[:50])

# Calculate accuracy
print(accuracy_score(y_test, predictions))
