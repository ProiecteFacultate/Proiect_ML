import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import cv2
import numpy as np

def setup():
    global data_directory, train_images, train_csv, val_csv, test_csv, train_labels, val_labels
    data_directory = "D:\Facultate\II\S_2\IA\Proiect_ML\Date_Proiect/"  # unde am salvate fisierele cu imagini si csvurile

    train_csv = pd.read_csv(data_directory + "train.csv")
    val_csv = pd.read_csv(data_directory + "val.csv")
    test_csv = pd.read_csv(data_directory + "test.csv")

    train_labels = train_csv["Class"]
    val_labels = val_csv["Class"]


def value_to_bin(x, bins):
    x = np.digitize(x, bins)
    return x - 1

setup()
train_features = []

for image_file in train_csv["Image"]:
    image = cv2.imread(data_directory + "train_images/" + image_file)
    image_as_array = np.array(image)
    train_features.append(image_as_array.flatten())

bins = np.linspace(0, 255 + 1, num=16 + 1)
X_train = value_to_bin(train_features, bins)

val_features = []
for image_file in val_csv["Image"]:
    image = cv2.imread(data_directory + "val_images/" + image_file)
    image_as_array = np.array(image)
    val_features.append(image_as_array.flatten())

bins = np.linspace(0, 255 + 1, num=16 + 1)
X_val = value_to_bin(val_features, bins)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, train_labels)

val_predictions = naive_bayes_model.predict(X_val)
accuracy = accuracy_score(val_labels, val_predictions)
print("Validation accuracy:", accuracy)

test_features = []
for image_file in test_csv["Image"]:
    image = cv2.imread(data_directory + "test_images/" + image_file)
    image_as_array = np.array(image)
    test_features.append(image_as_array.flatten())

bins = np.linspace(0, 255 + 1, num=16 + 1)
X_test = value_to_bin(test_features, bins)

test_predictions = naive_bayes_model.predict(X_test)

submission_df = pd.DataFrame({"Image": test_csv["Image"], "Class": test_predictions})
submission_df.to_csv(data_directory + "submission.csv", index=False)

