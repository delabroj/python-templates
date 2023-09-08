import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

data = pd.read_csv("DOGE-USD.csv")
print(data.head())

# Parse date
data["Date"] = pd.to_datetime(data["Date"], infer_datetime_format=True)
data.set_index("Date", inplace=True)

# View the correlation
print(data.corr())

# Check for null values
print(data.isnull().any())
print(data.isnull().sum())

# Drop rows with null values
data = data.dropna()

# View data statistics
print(data.describe())

# Plot close vs date
plt.figure(figsize=(20, 7))
x = data.groupby("Date")["Close"].mean()
x.plot(linewidth=2.5, color="b")
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Date vs Close")
plt.show()

# Add custom features
data["gap"] = (data["High"] - data["Low"]) * data["Volume"]
data["y"] = data["High"] / data["Volume"]
data["z"] = data["Low"] / data["Volume"]
data["a"] = data["High"] / data["Low"]
data["b"] = (data["High"] / data["Low"]) * data["Volume"]
print(abs(data.corr()["Close"].sort_values(ascending=False)))

# Take subset of features
data = data[["Close", "Volume", "gap", "a", "b"]]
print(data.head())

# Separate data into train and test sets
dataLength = 30
df2 = data.tail(dataLength)
trainingLength = 11
train = df2[:trainingLength]
test = df2[(trainingLength - dataLength) :]

print(train.shape, test.shape)

model = SARIMAX(endog=train["Close"], exog=train.drop("Close", axis=1), order=(2, 1, 1))
results = model.fit()
print(results.summary())

# Make prediction
start = trainingLength
end = dataLength - 1
predictions = results.predict(start=start, end=end, exog=test.drop("Close", axis=1))
print(predictions)

# Print actual set
plt.figure(1)
df2["Close"].plot(label="real", legend=True, figsize=(12, 6))

# Print prediction vs actual
plt.figure(2)
train["Close"].plot(label="train", legend=True, figsize=(12, 6))
test["Close"].plot(label="test", legend=True)
predictions.plot(label="prediction", legend=True)
plt.show()
