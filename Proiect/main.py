import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, normalize
import cv2
import numpy as np
from PIL import Image

def setup():
    global data_directory, train_csv, val_csv, test_csv
    data_directory = "D:\Facultate\II\S_2\IA\Proiect_ML\Date_Proiect/"  # unde am salvate fisierele cu imagini si csvurile
    train_csv = pd.read_csv(data_directory + "train.csv")
    val_csv = pd.read_csv(data_directory + "val.csv")
    test_csv = pd.read_csv(data_directory + "test.csv")

def extract_features(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_bins = 9
    s_bins = 9
    v_bins = 9
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    features = hist.flatten()
    return features

setup()
train_features = []
for image_file in train_csv["Image"]:
    features = extract_features(data_directory + "train_images/" + image_file)
    train_features.append(features)

val_features = []
for image_file in val_csv["Image"]:
    features = extract_features(data_directory + "val_images/" + image_file)
    val_features.append(features)

# Convert labels to numeric format
train_labels = train_csv["Class"]
val_labels = val_csv["Class"]

# Scale the features to a non-negative range
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_features)
X_val = scaler.transform(val_features)

# Initialize the Multinomial Naive Bayes model and fit it on the training data
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, train_labels)

# Evaluate the model performance on the validation data
val_predictions = naive_bayes_model.predict(X_val)
accuracy = accuracy_score(val_labels, val_predictions)
print("Validation accuracy:", accuracy)

# Extract features for the test set and scale them
test_features = []
for image_file in test_csv["Image"]:
    features = extract_features(data_directory + "test_images/" + image_file)
    test_features.append(features)

X_test = scaler.transform(test_features)

# Make predictions on the test set
test_predictions = naive_bayes_model.predict(X_test)

submission_df = pd.DataFrame({"Image": test_csv["Image"], "Class": test_predictions})
submission_df.to_csv(data_directory + "submission.csv", index=False)

