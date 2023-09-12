import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import TextVectorization
from keras import layers
import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow_hub as hub

tf.get_logger().setLevel("INFO")

# Get data
df = pd.read_csv("./spam.csv", encoding="latin-1")
print(df.head())
print(df.describe())

# Remove empty fields
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# Rename main columns
df = df.rename(columns={"v1": "label", "v2": "Text"})

# Encode label to numbers
df["label_enc"] = df["label"].map({"ham": 0, "spam": 1})

print(df.head())

# Plot ham vs spam count
# sns.countplot(x=df["label"])
# plt.show()


# Find average number of tokens in all sentences
avg_words_len = round(sum([len(i.split()) for i in df["Text"]]) / len(df["Text"]))
print(avg_words_len)


# Find number of unique words
s = set()
for sent in df["Text"]:
    for word in sent.split():
        s.add(word)
total_words_length = len(s)
print(total_words_length)

# Split data into train and test sets
X, y = np.asanyarray(df["Text"]), np.asanyarray(df["label_enc"])
new_df = pd.DataFrame({"Text": X, "label": y})
X_train, X_test, y_train, y_test = train_test_split(
    new_df["Text"], new_df["label"], test_size=0.2, random_state=42
)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# Helper functions for compiling, fitting, and evaluating the model performance
def compile_model(model):
    """
    simply compile the model with adam optimzer
    """
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )


def fit_model(
    model, epochs, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
):
    """
    fit the model with given epochs, train
    and test data
    """
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        validation_steps=int(0.2 * len(X_test)),
    )
    return history


def evaluate_model(model_name, index, model, X, y):
    """
    evaluate the model and returns accuracy,
    precision, recall and f1-score
    """
    y_preds = np.round(model.predict(X))
    accuracy = accuracy_score(y, y_preds)
    precision = precision_score(y, y_preds)
    recall = recall_score(y, y_preds)
    f1 = f1_score(y, y_preds)

    model_results_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
    }

    plt.figure(index)
    plt.title(model_name)
    cm = confusion_matrix(y_test, y_preds)
    conf_matrix = pd.DataFrame(
        data=cm, columns=["Predicted:0", "Predicted:1"], index=["Actual:0", "Actual:1"]
    )
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
    )

    return model_results_dict


# Model 0 - Naive Bayes
tfidf_vec = TfidfVectorizer().fit(X_train)
X_train_vec, X_test_vec = tfidf_vec.transform(X_train), tfidf_vec.transform(X_test)

baseline_model = MultinomialNB()
baseline_model.fit(X_train_vec, y_train)

# Model 1 - text vectorization and word embedding

# Create text vectorization
MAXTOKENS = total_words_length
OUTPUTLEN = avg_words_len

text_vec = TextVectorization(
    max_tokens=MAXTOKENS,
    standardize="lower_and_strip_punctuation",
    output_mode="int",
    output_sequence_length=OUTPUTLEN,
)
text_vec.adapt(X_train)

# Show text vectorization with sample sentence
sample_sentence = "This is a message"
print(text_vec([sample_sentence]))

# Create an embedding layer
embedding_layer = layers.Embedding(
    input_dim=MAXTOKENS,  # Size of the vocabulary
    output_dim=128,  # Dimension of the imbedding layer (the size of the vector in which the words will be embedded)
    embeddings_initializer="uniform",
    input_length=OUTPUTLEN,  # Length of the input sequences
)

# Build and comple model using Tensorflow Functional API
## Input layer
input_layer = layers.Input(shape=(1,), dtype=tf.string)
vec_layer = text_vec(input_layer)
embedding_layer_model = embedding_layer(vec_layer)
x = layers.GlobalAveragePooling1D()(embedding_layer_model)
x = layers.Flatten()(x)

## Hidden layer
x = layers.Dense(32, activation="relu")(x)

## Output layer
output_layer = layers.Dense(1, activation="sigmoid")(x)

model_1 = keras.Model(input_layer, output_layer)

model_1.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(label_smoothing=0.5),
    metrics=["accuracy"],
)

# Show model summary
print(model_1.summary())

# Train the model
history_1 = model_1.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_test, y_test),
    validation_steps=int(0.2 * len(X_test)),
)

# Show trainig history
# pd.DataFrame(history_1.history).plot()
# plt.show()


# Model -2 Bidirectional LSTM

# A bidirectional LSTM (Long short-term memory) is made up of two LSTMs, one accepting input in one direction and the other in the other.
# BiLSTMs effectively improve the network’s accessible information, boosting the context for the algorithm (e.g. knowing what words immediately follow and precede a word in a sentence).

input_layer = layers.Input(shape=(1,), dtype=tf.string)
vec_layer = text_vec(input_layer)
embedding_layer_model = embedding_layer(vec_layer)
bi_lstm = layers.Bidirectional(
    layers.LSTM(64, activation="tanh", return_sequences=True)
)(embedding_layer_model)
lstm = layers.Bidirectional(layers.LSTM(64))(bi_lstm)
flatten = layers.Flatten()(lstm)
dropout = layers.Dropout(0.1)(flatten)
x = layers.Dense(32, activation="relu")(dropout)
output_layer = layers.Dense(1, activation="sigmoid")(x)
model_2 = keras.Model(input_layer, output_layer)

compile_model(model_2)  # compile the model
history_2 = fit_model(model_2, epochs=5)  # fit the model


# Model -3 Transfer Learning with USE Encoder

# Transfer Learning
# Transfer learning is a machine learning approach in which a model generated for one job is utilized as the foundation for a model on a different task.
# USE Layer (Universal Sentence Encoder)
# The Universal Sentence Encoder converts text into high-dimensional vectors that may be used for text categorization, semantic similarity, and other natural language applications.
# The USE can be downloaded from tensorflow_hub and can be used as a layer using .kerasLayer() function.


# model with Sequential api
model_3 = keras.Sequential()

# universal-sentence-encoder layer
# directly from tfhub
use_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    trainable=False,
    input_shape=[],
    dtype=tf.string,
    name="USE",
)
model_3.add(use_layer)
model_3.add(layers.Dropout(0.2))
model_3.add(layers.Dense(64, activation=keras.activations.relu))
model_3.add(layers.Dense(1, activation=keras.activations.sigmoid))

compile_model(model_3)

history_3 = fit_model(model_3, epochs=20)


# Compare results from different models
baseline_model_results = evaluate_model(
    "MultinomialNB Model", 1, baseline_model, X_test_vec, y_test
)
model_1_results = evaluate_model(
    "Custom-Vec-Embedding Model", 2, model_1, X_test, y_test
)
model_2_results = evaluate_model("Bidirectional-LSTM Model", 3, model_2, X_test, y_test)
model_3_results = evaluate_model(
    "USE-Transfer learning Model", 4, model_3, X_test, y_test
)

total_results = pd.DataFrame(
    {
        "MultinomialNB Model": baseline_model_results,
        "Custom-Vec-Embedding Model": model_1_results,
        "Bidirectional-LSTM Model": model_2_results,
        "USE-Transfer learning Model": model_3_results,
    }
).transpose()

print(total_results)

plt.show()
