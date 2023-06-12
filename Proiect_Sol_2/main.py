import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler



class Knn_classifier:
    def __init__(self, train_images, train_labels):
        # Ex. 1
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors, metric='l2'):
        if metric == 'l2':    #euclidian
            distances = np.sqrt(np.sum(np.square(self.train_images - test_image), axis=(1, 2, 3)))
        elif metric == 'l1':  #manhaattan
            distances = np.sum(np.abs(self.train_images - test_image), axis=(1, 2, 3))
        else:
            raise ValueError()

        best_indices = np.argsort(distances)
        top_k_indices = best_indices[:num_neighbors].tolist()
        tok_k_labels = np.array(self.train_labels)[top_k_indices].ravel()
        counts = np.bincount(tok_k_labels)
        pred_label = np.argmax(counts)

        return pred_label

    def classify_images(self, test_images, num_neighbors, metric='l2'):
        num_test_images = test_images.shape[0]
        predicted_labels = np.zeros((num_test_images), np.int8)

        for i in range(num_test_images):
            predicted_labels[i] = self.classify_image(test_images[i, :], num_neighbors=num_neighbors, metric=metric)

        return predicted_labels

    def accuracy_score(self, y_pred, y_true):
        return np.sum(y_pred == y_true) / len(y_pred)


def setup():
    global data_directory, train_images, train_csv, val_csv, test_csv, train_labels, val_labels
    data_directory = "D:\Facultate\II\S_2\IA\Proiect_ML\Date_Proiect/"  # unde am salvate fisierele cu imagini si csvurile

    train_csv = pd.read_csv(data_directory + "train.csv")
    val_csv = pd.read_csv(data_directory + "val.csv")
    test_csv = pd.read_csv(data_directory + "test.csv")

    train_labels = train_csv["Class"]
    val_labels = val_csv["Class"]


setup()
train_images = []
train_labels = []
scaler = MinMaxScaler()  # Create a MinMaxScaler instance for normalization
for image_file, label in zip(train_csv["Image"], train_csv["Class"]):
    image = cv2.imread(data_directory + "train_images/" + image_file)
    image_as_array = np.array(image)
    train_images.append(image_as_array)
    train_labels.append(label)

knn_classifier = Knn_classifier(train_images=train_images, train_labels=train_labels)

val_images = []
val_labels = []
for image_file, label in zip(val_csv["Image"], val_csv["Class"]):
    image = cv2.imread(data_directory + "val_images/" + image_file)
    image_as_array = np.array(image)
    val_images.append(image_as_array)
    val_labels.append(label)

test_images = []
for image_file in test_csv["Image"]:
    image = cv2.imread(data_directory + "test_images/" + image_file)
    image_as_array = np.array(image)
    test_images.append(image_as_array)

val_images = np.array(val_images)    #altfel e list in loc de numpyarray si da eroarea ca nu are attribute shape
val_predictions = knn_classifier.classify_images(val_images, num_neighbors=498, metric='l2')
acc_score = knn_classifier.accuracy_score(val_predictions, val_labels)
print("Accuracy score: " + str(acc_score))

test_images = np.array(test_images)
test_predictions = knn_classifier.classify_images(test_images, num_neighbors=81, metric='l2')
submission_df = pd.DataFrame({"Image": test_csv["Image"], "Class": test_predictions})
submission_df.to_csv(data_directory + "submission.csv", index=False)