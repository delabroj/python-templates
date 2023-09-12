# Importing necessary libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.metrics import confusion_matrix

nltk.download("stopwords")

# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from keras.preprocessing.text import Tokenizer

from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load data
data = pd.read_csv("emails.csv")
print(data.head())

# View data size
print(data.shape)

# Plot spam vs ham
sns.countplot(x="spam", data=data)
plt.show()

# Downsample the ham so the data is balanced
# Downsampling to balance the dataset
ham_msg = data[data.spam == 0]
spam_msg = data[data.spam == 1]
ham_msg = ham_msg.sample(n=len(spam_msg), random_state=42)

balanced_data = pd.concat([ham_msg, spam_msg]).reset_index(drop=True)
plt.figure(figsize=(8, 6))
sns.countplot(data=balanced_data, x="spam")
plt.show()

# Get rid of 'Subject'
balanced_data["text"] = balanced_data["text"].str.replace("Subject", "")
print(balanced_data.head())

# Remove punctuation
punctuations_list = string.punctuation


def remove_punctuations(text):
    temp = str.maketrans("", "", punctuations_list)
    return text.translate(temp)


balanced_data["text"] = balanced_data["text"].apply(lambda x: remove_punctuations(x))
print(balanced_data.head())


# Remove stop words
def remove_stopwords(text):
    stop_words = stopwords.words("english")

    imp_words = []

    # Storing the important words
    for word in str(text).split():
        word = word.lower()

        if word not in stop_words:
            imp_words.append(word)

    output = " ".join(imp_words)

    return output


balanced_data["text"] = balanced_data["text"].apply(lambda text: remove_stopwords(text))
print(balanced_data.head())


# Create word clouds
def plot_word_cloud(data, typ):
    email_corpus = " ".join(data["text"])

    wc = WordCloud(
        background_color="black",
        max_words=100,
        width=800,
        height=400,
        collocations=False,
    ).generate(email_corpus)

    plt.imshow(wc, interpolation="bilinear")
    plt.title(f"WordCloud for {typ} emails", fontsize=15)
    plt.axis("off")


plt.figure(1)
plot_word_cloud(balanced_data[balanced_data["spam"] == 0], typ="Non-Spam")
plt.figure(2)
plot_word_cloud(balanced_data[balanced_data["spam"] == 1], typ="Spam")
plt.show()

# Split into train and test sets
# train test split
X_train, X_test, y_train, y_test = train_test_split(
    balanced_data["text"], balanced_data["spam"], test_size=0.2, random_state=42
)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences to have the same length
max_len = 100  # maximum sequence length
train_sequences = pad_sequences(
    train_sequences, maxlen=max_len, padding="post", truncating="post"
)
test_sequences = pad_sequences(
    test_sequences, maxlen=max_len, padding="post", truncating="post"
)

# Build the model
model = tf.keras.models.Sequential()

# Input layer
model.add(
    tf.keras.layers.Embedding(
        input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_len
    )
)

# LSTM layer to identify useful patterns in the sequence
model.add(tf.keras.layers.LSTM(16))

# One fully connected layer
model.add(tf.keras.layers.Dense(32, activation="relu"))

# Output layer
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

print(model.summary())

# While compiling a model we provide these three essential parameters:

# optimizer – This is the method that helps to optimize the cost function by using gradient descent.
# loss – The loss function by which we monitor whether the model is improving with training or not.
# metrics – This helps to evaluate the model by predicting the training and the validation data.
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
    optimizer="adam",
)

es = EarlyStopping(patience=3, monitor="val_accuracy", restore_best_weights=True)

lr = ReduceLROnPlateau(patience=2, monitor="val_loss", factor=0.5, verbose=0)

# Train the model
history = model.fit(
    train_sequences,
    y_train,
    validation_data=(test_sequences, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[lr, es],
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_sequences, y_test)
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_accuracy)

# Plot accuracy during training
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# Plot the confusion matrix

y_pred = model.predict(test_sequences)

cm = confusion_matrix(y_test, [round(y) for y in y_pred.flatten()])
conf_matrix = pd.DataFrame(
    data=cm, columns=["Predicted:0", "Predicted:1"], index=["Actual:0", "Actual:1"]
)

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens")

plt.show()
