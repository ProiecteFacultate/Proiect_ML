import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Setările pentru dimensiunea imaginilor
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

# Calea către fișierele de date
data_directory = "D:\Facultate\II\S_2\IA\Proiect_ML\Date_Proiect/"

# Încarcă datele de antrenare, validare și test
train_df = pd.read_csv(data_directory + "train.csv")
val_df = pd.read_csv(data_directory + "val.csv")
test_df = pd.read_csv(data_directory + "test.csv")

# Preprocesarea datelor
le = LabelEncoder()
train_labels = le.fit_transform(train_df["Class"])
val_labels = le.transform(val_df["Class"])

# Funcție pentru încărcarea și redimensionarea imaginilor
def load_and_preprocess_image(image_file):
    image = cv2.imread(image_file)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image / 255.0  # Normalizare
    return image

# Procesarea imaginilor de antrenare
train_images = np.array([load_and_preprocess_image(data_directory + "train_images/" + image_file) for image_file in train_df["Image"]])

# Procesarea imaginilor de validare
val_images = np.array([load_and_preprocess_image(data_directory + "val_images/" + image_file) for image_file in val_df["Image"]])

# Procesarea imaginilor de test
test_images = np.array([load_and_preprocess_image(data_directory + "test_images/" + image_file) for image_file in test_df["Image"]])

# Definirea arhitecturii modelului CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(96, activation="softmax"))

# Compilarea modelului
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])

# Antrenarea modelului
model.fit(train_images, train_labels, batch_size=32, epochs=10, validation_data=(val_images, val_labels))

# Evaluarea modelului pe setul de test
test_predictions = np.argmax(model.predict(test_images), axis=-1)

# Salvarea rezultatelor într-un fișier de submisie
submission_df = pd.DataFrame({"Image": test_df["Image"], "Class": le.inverse_transform(test_predictions)})
submission_df.to_csv(data_directory + "submission.csv", index=False)
