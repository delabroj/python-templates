import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the training data
data = pd.read_csv("Training.csv")

# Drop the last column (which is empty)
data = data.dropna(axis=1)

print(data.head())

# Check if the dataset is balanced
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame(
    {"Disease": disease_counts.index, "Counts": disease_counts.values}
)

plt.figure(figsize=(18, 8))
sns.barplot(x="Disease", y="Counts", data=temp_df)
plt.xticks(rotation=90)
plt.show()

# Each disease has 120 data points, so the training set is balanced


# Convert prognosis column to integers
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Split data into training and validation sets
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=24
)

print(f"Train: {X_train.shape}, {Y_train.shape}")
print(f"Test: {X_test.shape}, {Y_test.shape}")


# Define scoring metric for k-fold cross validation
def cv_scoring(estimator, x, y):
    return accuracy_score(y, estimator.predict(x))


# Initialize models
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18),
}

# Produce cross validation score for models
for model_name, model in models.items():
    scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring=cv_scoring)

    print("==" * 30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")

# Train SVM classifier
svm_model = SVC()
svm_model.fit(X_train, Y_train)

# Test
predictions = svm_model.predict(X_test)
print(
    f"Accuracy on train data by SVM Classifier: {accuracy_score(Y_train, svm_model.predict(X_train))*100}"
)
print(
    f"Accuracy on test data by SVM Classifier: {accuracy_score(Y_test, predictions)*100}"
)

cf_matrix = confusion_matrix(Y_test, predictions)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM classifier on test data")
plt.show()

# Train Naive Bayes classifier
nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)

# Test
predictions = nb_model.predict(X_test)
print(
    f"Accuracy on train data by Naive Bayes Classifier: {accuracy_score(Y_train, nb_model.predict(X_train))*100}"
)
print(
    f"Accuracy on test data by Naive Bayes Classifier: {accuracy_score(Y_test, predictions)*100}"
)

cf_matrix = confusion_matrix(Y_test, predictions)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Naive Bayes classifier on test data")
plt.show()

# Train Random Forest classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)

# Test
predictions = rf_model.predict(X_test)
print(
    f"Accuracy on train data by Random Forest Classifier: {accuracy_score(Y_train, rf_model.predict(X_train))*100}"
)
print(
    f"Accuracy on test data by Random Forest Classifier: {accuracy_score(Y_test, predictions)*100}"
)

cf_matrix = confusion_matrix(Y_test, predictions)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest classifier on test data")
plt.show()

# Now train models on the whole training set
final_svm_model = SVC(probability=True)
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X.values, Y)
final_nb_model.fit(X.values, Y)
final_rf_model.fit(X.values, Y)

# Read the test data
test_data = pd.read_csv("./Testing.csv").dropna(axis=1)

test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])

# Combine models by taking the mode of their predictions
svm_preds = final_svm_model.predict(test_X.values)
nb_preds = final_nb_model.predict(test_X.values)
rf_preds = final_rf_model.predict(test_X.values)

final_preds = [mode([i, j, k])[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

print(
    f"Accuracy on Test dataset by combined model: {accuracy_score(test_Y, final_preds)*100}"
)

cf_matrix = confusion_matrix(test_Y, final_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion matrix for combined model on test dataset")
plt.show()

# Create function which takes list of symptoms and returns a likely disease
symptoms = X.columns.values

# Encode the input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {"symptom_index": symptom_index, "predictions_classes": encoder.classes_}


# Input: string containing symptoms separated by commas
def predictDisease(simptoms_):
    simptoms_ = simptoms_.split(",")

    input_data = [0] * len(data_dict["symptom_index"])
    for symptom_ in simptoms_:
        index_ = data_dict["symptom_index"][symptom_]
        input_data[index_] = 1

    # Reshape data for model input
    input_data = np.array(input_data).reshape(1, -1)

    # Get individual predictions
    rf_prediction = final_rf_model.predict(input_data)[0]
    nb_prediction = final_nb_model.predict(input_data)[0]
    svm_prediction = final_svm_model.predict(input_data)[0]

    # Make final prediction
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0]

    return {
        "rf_model_prediction": data_dict["predictions_classes"][rf_prediction],
        "rf_model_prediction_confidence": final_rf_model.predict_proba(input_data)[0][
            rf_prediction
        ],
        "nb_model_prediction": data_dict["predictions_classes"][nb_prediction],
        "nb_model_prediction_confidence": final_nb_model.predict_proba(input_data)[0][
            nb_prediction
        ],
        "svm_model_prediction": data_dict["predictions_classes"][svm_prediction],
        "svm_model_prediction_confidence": final_svm_model.predict_proba(input_data)[0][
            svm_prediction
        ],
        "final_prediction": data_dict["predictions_classes"][final_prediction],
    }


# Test final function
print(predictDisease("Skin Rash,Joint Pain,Acidity,Vomiting,Fatigue"))
