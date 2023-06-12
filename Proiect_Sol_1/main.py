import pandas as pd
from pandas import read_csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
from PIL import Image

def deschiderInitialaDeFisiere():
    global directorCuDate, CSVdeAntrenare, CSVdeValidare, CSVdeTest, listaClaseCsvDeAntrenament, listaClaseCsvDeValidare
    directorCuDate = "D:\Facultate\II\S_2\IA\Proiect_ML\Date_Proiect/"  # unde am salvate fisierele cu imagini si csvurile la mine in pc

    CSVdeAntrenare = read_csv(directorCuDate + "train.csv")
    CSVdeValidare = read_csv(directorCuDate + "val.csv")
    CSVdeTest = read_csv(directorCuDate + "test.csv")

    listaClaseCsvDeAntrenament = CSVdeAntrenare["Class"]
    listaClaseCsvDeValidare = CSVdeValidare["Class"]


def transformaListaInValoriDiscrete(lista, nrIntervale):
    listaDupaTransformare = np.digitize(lista, nrIntervale)  #primeste o lista de valori si le discretizeaza (inlocuieste o valoare cu intervalul din care face parte)
    listaIndexataDeLaZero = listaDupaTransformare - 1   #digitise are indexarea de la 1 si facem -1 ca sa avem indexare de la 0
    return listaIndexataDeLaZero

deschiderInitialaDeFisiere()
valoareMaximaInterval = 256  #255 e val maxima pt ca pixelii au val maxima 255 si facem +1 pt ca e deschis la capete
nrIntervale = 17   #am ales 16 pt ca am vazut ca da cel mai bun rezultat si am adaugat 1 pt ca deschis la capete
capeteIntervale = np.linspace(0, valoareMaximaInterval, num=nrIntervale)

#prelucram datele pt imaginile de antrenare
trasaturiDateDeAntrenament = []
listaImaginiCsvDeAntrenament = CSVdeAntrenare["Image"]  #o lista cu numele imaginilor din csvul de antrenament
for i in range(len(listaImaginiCsvDeAntrenament)):
    numeImagine = listaImaginiCsvDeAntrenament[i]
    imagine = Image.open(directorCuDate + "train_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)         #vrem sa avem datele legate de pixeli sub forma unui array din numpy
    trasaturiDateDeAntrenament.append(imagineCaNpArray.flatten())     #folosim flateen ca sa facem arrayul sa fie 1D

valoriPrelucrateAntrenament = transformaListaInValoriDiscrete(trasaturiDateDeAntrenament, capeteIntervale)

#prelucram datele pt imaginile de validare
trasaturiDateDeValidare = []
listaImaginiCsvDeValidare = CSVdeValidare["Image"]  #o lista cu numele imaginilor din csvul de validare
for i in range(len(listaImaginiCsvDeValidare)):
    numeImagine = listaImaginiCsvDeValidare[i]
    imagine = Image.open(directorCuDate + "val_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)
    trasaturiDateDeValidare.append(imagineCaNpArray.flatten())

valoriPrelucrateValidare = transformaListaInValoriDiscrete(trasaturiDateDeValidare, capeteIntervale)

modelNaiveBayes = MultinomialNB()
modelNaiveBayes.fit(valoriPrelucrateAntrenament, listaClaseCsvDeAntrenament)

predictiiValidare = modelNaiveBayes.predict(valoriPrelucrateValidare)
acuratete = accuracy_score(listaClaseCsvDeValidare, predictiiValidare)
print("Acuratete de validare: " + str(acuratete))

#prelucram datele pt imaginile de test
trasaturiDateDeTest = []
listaImaginiCsvDeTest = CSVdeTest["Image"]  #o lista cu numele imaginilor din csvul de test
for i in range(len(listaImaginiCsvDeTest)):
    numeImagine = listaImaginiCsvDeTest[i]
    imagine = Image.open(directorCuDate + "test_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)
    trasaturiDateDeTest.append(imagineCaNpArray.flatten())

valoriPrelucrateTest = transformaListaInValoriDiscrete(trasaturiDateDeTest, capeteIntervale)

predictiiTest = modelNaiveBayes.predict(valoriPrelucrateTest)

fisierDePredictii = pd.DataFrame({"Image": CSVdeTest["Image"], "Class": predictiiTest})  #face fisierul unde pune clasa care a fost prezisa pt fiecare imagine
fisierDePredictii.to_csv(directorCuDate + "submission.csv", index=False)

