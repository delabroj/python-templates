import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

ipl = pd.read_csv("ipl_dataset.csv")
print(ipl.head())

data = pd.read_csv("IPL Player Stats - 2016 till 2019.csv")
print(data.head())

# Merge data sets
ipl = ipl.drop(["Unnamed: 0", "extras", "match_id", "runs_off_bat"], axis=1)
new_ipl = pd.merge(ipl, data, left_on="striker", right_on="Player", how="left")
new_ipl.drop(["wicket_type", "player_dismissed"], axis=1, inplace=True)
print(new_ipl.columns)

# Show y statistics
new_ipl["y"].hist(bins=20, figsize=(10, 10))
plt.show()

# Fill null values
str_cols = new_ipl.columns[new_ipl.dtypes == object]
new_ipl[str_cols] = new_ipl[str_cols].fillna(".")

# Show columns containing object type
for c in new_ipl.columns:
    if new_ipl[c].dtype == object:
        print(c, "->", new_ipl[c].dtype)

# Encode string values as numbers
a1 = new_ipl["venue"].unique()
a2 = new_ipl["batting_team"].unique()
a3 = new_ipl["bowling_team"].unique()
a4 = new_ipl["striker"].unique()
a5 = new_ipl["bowler"].unique()


def labelEncoding(data):
    dataset = pd.DataFrame(data)
    feature_dict = {}

    for feature in dataset:
        if dataset[feature].dtype == object:
            labelEncoder = preprocessing.LabelEncoder()
            fs = dataset[feature].unique()
            labelEncoder.fit(fs)
            dataset[feature] = labelEncoder.transform(dataset[feature])
            feature_dict[feature] = labelEncoder

    return dataset


new_ipl = labelEncoding(new_ipl)

ip_dataset = new_ipl[
    [
        "venue",
        "innings",
        "batting_team",
        "bowling_team",
        "striker",
        "non_striker",
        "bowler",
    ]
]

b1 = ip_dataset["venue"].unique()
b2 = ip_dataset["batting_team"].unique()
b3 = ip_dataset["bowling_team"].unique()
b4 = ip_dataset["striker"].unique()
b5 = ip_dataset["bowler"].unique()
new_ipl.fillna(0, inplace=True)

features = {}

for i in range(len(a1)):
    features[a1[i]] = b1[i]
for i in range(len(a2)):
    features[a2[i]] = b2[i]
for i in range(len(a3)):
    features[a3[i]] = b3[i]
for i in range(len(a4)):
    features[a4[i]] = b4[i]
for i in range(len(a5)):
    features[a5[i]] = b5[i]

print(features)

print(new_ipl.head())
print(new_ipl.y.describe())

# Select features we will use
X = new_ipl.drop("y", axis=1).values
y = new_ipl["y"].values


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Scale values
scaler = preprocessing.MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build and train model
model = Sequential()

model.add(Dense(43, activation="relu"))
# model.add(Dropout(0.5))

model.add(Dense(86, activation="relu"))
# model.add(Dropout(0.5))

# model.add(Dense(11, activation="relu"))
# model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

# Early stopping is done to avoid overfitting.

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=25,
    verbose=1,
    mode="min",
)

model.fit(
    x=X_train,
    y=y_train,
    epochs=400,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1,
)

# Plot loss functions by epoch
model_losses = pd.DataFrame(model.history.history)
model_losses.plot()
plt.show()

# Make predictions and compare to actual values
predictions = model.predict(X_test)
sample = pd.DataFrame(predictions, columns=["Predict"])
sample["Actual"] = y_test
print(sample.head(100))


print("mean absolute error: ", mean_absolute_error(y_test, predictions))

print("sqrt of mean squared error: ", np.sqrt(mean_squared_error(y_test, predictions)))


plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()


# Compare to simply predicting the mean of all historical scores
predictions.fill(51.431771)
print(predictions)


print("mean absolute error: ", mean_absolute_error(y_test, predictions))

print("sqrt of mean squared error: ", np.sqrt(mean_squared_error(y_test, predictions)))
