import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib

data = pd.read_csv("creditcard.csv")

# Print the first 5 rows
print(data.head())

# Show shape of data
print(data.shape)

# Show data stats
print(data.describe())

# Determine percent of cases that are fraud
fraud = data[data["Class"] == 1]
valid = data[data["Class"] == 0]
fraudFraction = len(fraud) / float(len(valid))

print(f"Fraud Cases: {len(fraud)}")
print(f"Valid Cases: {len(valid)}")
print(f"Percent fraudulent: {fraudFraction * 100}%")

# Show amount stats for fraud transactions
print("Fraud amount stats")
print(fraud.Amount.describe())

# Show amount stats for valid transactions
print("Valid amount stats")
print(valid.Amount.describe())

# Plot correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

# Separate x and y data
X = data.drop(["Class"], axis=1)
Y = data["Class"]
print(X.shape)
print(Y.shape)

# Get just the values for the processesing steps
xData = X.values
yData = Y.values

# Split data into training and test sets
xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.2, random_state=42
)

# Build a random forest classifier
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

# Save model weights
joblib.dump(rfc, "./service/model.bin", compress=3)

# Find predictions from test set
yPred = rfc.predict(xTest)

# Show classifier scores
n_outliers = len(fraud)
n_errors = (yPred != yTest).sum()

acc = metrics.accuracy_score(yTest, yPred)
print(f"The accuracy is {acc}")

prec = metrics.precision_score(yTest, yPred)
print(f"The precision is {prec}")

rec = metrics.recall_score(yTest, yPred)
print(f"The recall is {rec}")

f1 = metrics.f1_score(yTest, yPred)
print(f"The F1-score is {f1}")

MCC = metrics.matthews_corrcoef(yTest, yPred)
print(f"The Matthews correlation coefficient is {MCC}")

# Plot the confusion matrix
labels = ["Normal", "Fraud"]
conf_matrix = metrics.confusion_matrix(yTest, yPred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()
