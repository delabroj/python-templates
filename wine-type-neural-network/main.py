import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import tensorflow as tf

# Read in wine data
white = pd.read_csv("./winequality-white.csv", sep=";")
red = pd.read_csv("./winequality-red.csv", sep=";")

print(white.sample(5))
print(red.tail())

# Check for null values
print(pd.isnull(white).describe())
print(pd.isnull(red).describe())

# Alcohol content histograms
fig, ax = plt.subplots(1, 2)

ax[0].hist(red.alcohol, 10, facecolor="red", alpha=0.5, label="Red wine")

ax[1].hist(
    white.alcohol,
    10,
    facecolor="white",
    ec="black",
    lw=0.5,
    alpha=0.5,
    label="White wine",
)

ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_ylim([0, 1000])
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")

fig.suptitle("Distribution of Alcohol in % Vol")
plt.show()

# Add type columns
red["type"] = 1
white["type"] = 0

wines = pd.concat([red, white], ignore_index=True)

X = wines.iloc[:, :-1]
y = np.ravel(wines.type)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize model constructor
model = Sequential()

# Add input layer
model.add(Dense(12, activation="relu", input_shape=(12,)))

# Add one hidden layer
model.add(Dense(6, activation="relu"))

# Add output layer
model.add(Dense(1, activation="sigmoid"))

print("Model output shape: ", model.output_shape)
print("Model summary: ", model.summary())
print("Model config: ", model.get_config())

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
print(X_train.shape)
print(y_train.shape)
model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=1)

print("Model weight tensors: ", model.get_weights())


# Run some predictions
y_pred = model.predict(X_test)

y_pred = y_pred.flatten()
y_pred = np.array([round(v, 0) for v in y_pred])
print(y_pred.shape)
print(y_test.shape)

# Print full numpy arrays
np.set_printoptions(threshold=sys.maxsize)
print(y_pred)
print(y_test)

cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(
    data=cm, columns=["Predicted:0", "Predicted:1"], index=["Actual:0", "Actual:1"]
)

print("The details for confusion matrix is =")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens")

plt.show()

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96,
    layer_range=None,
    show_layer_activations=True,
    show_trainable=True,
)
