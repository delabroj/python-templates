import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    jaccard_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

df = pd.read_csv("./heart.csv")

# Drop column
df.drop(["age"], inplace=True, axis=1)

# Remove rows with Na values
print(len(df))
df.dropna(axis=0, inplace=True)
print(len(df))

print(df.head(), df.shape)
print(df.target.value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(x="target", data=df, palette="BuGn_r")
plt.show()

X = df.drop(["target"], axis=1)
Y = df["target"]


# Normalize
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split to train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print("Train set:", X_train.shape, Y_train.shape)
print("Test set:", X_test.shape, Y_test.shape)

# Train logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

df["predicted_target"] = logreg.predict(X)

print(df.head(100))

# Evaluate accuracy
print("")
print(
    "Accuracy of the model in jaccard similarity score is = ",
    jaccard_score(Y_test, Y_pred),
)
print(
    "Accuracy score is = ",
    accuracy_score(Y_test, Y_pred),
)


# Plot the confusion matrix

cm = confusion_matrix(Y_test, Y_pred)
conf_matrix = pd.DataFrame(
    data=cm, columns=["Predicted:0", "Predicted:1"], index=["Actual:0", "Actual:1"]
)

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens")

plt.show()

print("The details for confusion matrix is =")
print(classification_report(Y_test, Y_pred))
