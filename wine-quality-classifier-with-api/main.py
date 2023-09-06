import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

df = pd.read_csv("winequality.csv")

# Print the first 5 rows
print(df.head())

# Show column information
df.info()

# Convert string 'type' column to numbers
df.replace({"white": 1, "red": 0}, inplace=True)

# Show column statistics
print(df.describe().T)

# Show number of null values in each column
print(df.isnull().sum())

# Set null values to the column mean
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# Verify that there are no more null values
print(df.isnull().sum().sum())

# Draw histograms of each column
df.hist(bins=20, figsize=(10, 10))
plt.show()

# Show quality by alcohol content
plt.bar(df["quality"], df["alcohol"])
plt.xlabel("quality")
plt.ylabel("alcohol")
plt.show()

# Show correlation heat map to determine if some features are redundant
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.6, annot=True, cbar=False)
plt.show()

# 'total sulfer dioxide' and 'free sulfer dioxide' appear to be correlated, so let's remove 'total sulfer dioxide'
df = df.drop("total sulfur dioxide", axis=1)

# Create a 'best quality' feature derived from the 'quality' feature
df["best quality"] = [1 if x > 5 else 0 for x in df.quality]

# Segregate features and target variables
features = df.drop(["quality", "best quality"], axis=1)
target = df["best quality"]

# Split data set into training and test sets 80:20
xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40
)

# Show size of training and testing sets
print(xtrain.shape, xtest.shape)

# Normalize data to make training faster and more stable
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

# Train 3 different machine learning models
models = [LogisticRegression(), XGBClassifier(), SVC(kernel="rbf")]

for _, model in enumerate(models):
    model.fit(xtrain, ytrain)

    print(f"{model} : ")
    print("Training Accuracy : ", metrics.roc_auc_score(ytrain, model.predict(xtrain)))
    print("Validation Accuracy : ", metrics.roc_auc_score(ytest, model.predict(xtest)))
    print()


# Plot confusion matrix for XGBClassifier
metrics.ConfusionMatrixDisplay.from_estimator(models[1], xtest, ytest, normalize="all")
plt.show()

# Print classification report for XGBClassifier
print(metrics.classification_report(ytest, models[1].predict(xtest)))


# Save XGBClassifier model
models[1].save_model("service/model.bin")
