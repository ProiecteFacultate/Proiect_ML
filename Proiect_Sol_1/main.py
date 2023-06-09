from pandas import read_csv
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


def deschiderInitialaDeFisiere():
    global directorCuDate, CSVdeAntrenare, CSVdeValidare, CSVdeTest, claseCsvDeAntrenament, claseCsvDeValidare, numeImaginiDeAntrenament, numeImaginiDeValidare, numeImaginiDeTest
    directorCuDate = "D:\Facultate\II\S_2\IA\Proiect_ML\Date_Proiect/"  # unde am salvate fisierele cu imagini si csvurile la mine in pc

    CSVdeAntrenare = read_csv(directorCuDate + "train.csv")
    CSVdeValidare = read_csv(directorCuDate + "val.csv")
    CSVdeTest = read_csv(directorCuDate + "test.csv")

    claseCsvDeAntrenament = CSVdeAntrenare["Class"]
    claseCsvDeValidare = CSVdeValidare["Class"]

    numeImaginiDeAntrenament = CSVdeAntrenare["Image"]  # o lista cu numele imaginilor din csvul de antrenament
    numeImaginiDeValidare = CSVdeValidare["Image"]  # o lista cu numele imaginilor din csvul de validare
    numeImaginiDeTest = CSVdeTest["Image"]  # o lista cu numele imaginilor din csvul de test


def transformaListaInValoriDiscrete(lista, nrIntervale):
    listaDupaTransformare = np.digitize(lista, nrIntervale)  #primeste o lista de valori si le discretizeaza (inlocuieste o valoare cu intervalul din care face parte)
    listaIndexataDeLaZero = listaDupaTransformare - 1   #digitise are indexarea de la 1 si facem -1 ca sa avem indexare de la 0
    return listaIndexataDeLaZero


def calculeazaAcuratete(clasePrezise, claseReale):
    numarPredictiiCorecte = 0
    numarImagini = len(clasePrezise)  # clasePrezise e o lista cu clasele prezise pt fiecare imagine deci nrImagini = nrClasePrezise = nrClaseReale

    for i in range(numarImagini):
        if clasePrezise[i] == claseReale[i]:
            numarPredictiiCorecte += 1

    acuratete = numarPredictiiCorecte / numarImagini
    return acuratete



def dateDespreAcuratete():   #se face matricea de confuzie, precision si recall
    #precision si recall pentru fiecare clasa

    for clasa in range(96):
        truePositive = falsePositive = falseNegative = trueNegative = 0
        for i in range(len(predictiiValidare)):
            if predictiiValidare[i] == clasa and claseCsvDeValidare[i] == clasa:
                truePositive += 1
            elif predictiiValidare[i] == clasa and claseCsvDeValidare[i] != clasa:
                falsePositive += 1
            elif predictiiValidare[i] != clasa and claseCsvDeValidare[i] == clasa:
                falseNegative += 1
            else:
                trueNegative += 1

        if truePositive + falsePositive != 0:  # daca nu facem verificarea se poate primi division by zero
            precizie = truePositive / (truePositive + falsePositive)
        else:
            precizie = 0

        if truePositive + falseNegative != 0:
            recall = truePositive / (truePositive + falseNegative)
        else:
            recall = 0

        print("Pentru clasa " + str(clasa) + " avem precision = " + str("{:.3f}".format(precizie)) + " si recall = " + str("{:.3f}".format(recall)))


    #matricea de confuzie
    matriceConfuzie = [[0 for j in range(96)] for i in range(96)]

    for i in range(len(predictiiValidare)):
        valoarePrezisa = int(predictiiValidare[i])
        valoareReala = int(claseCsvDeValidare[i])
        matriceConfuzie[valoareReala][valoarePrezisa] += 1

    print("Matricea de confuzie:")

    for i in range(len(matriceConfuzie)):
        for j in range(len(matriceConfuzie[i])):
            print(matriceConfuzie[i][j], end=" ")
        print('\n')

    # deseneazaMatrice(matriceConfuzie)

# def deseneazaMatrice(matrice):
#     plt.rcParams["figure.figsize"] = [50, 50]
#     plt.rcParams["figure.autolayout"] = True
#     fig, ax = plt.subplots()
#     ax.matshow(matrice, cmap='binary')
#     plt.show()


######################## MAAIIIIIIIIIIIINNN  #######################


deschiderInitialaDeFisiere()
modelNaiveBayes = MultinomialNB()
valoareMaximaInterval = 256  #255 e val maxima pt ca pixelii au val maxima 255 si facem +1 pt ca e deschis la capete
nrIntervale = 17   #am ales 17 pt ca am vazut ca da cel mai bun rezultat
capeteIntervale = np.linspace(0, valoareMaximaInterval, num=nrIntervale)   #generam nrIntervale valori egal distantate in intervalul [0, valoareMaximaInterval)

# prelucram datele pt imaginile de antrenare
trasaturiDateDeAntrenament = []
for i in range(len(numeImaginiDeAntrenament)):
    numeImagine = numeImaginiDeAntrenament[i]
    imagine = Image.open(directorCuDate + "train_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)  # vrem sa avem datele legate de pixeli sub forma unui array din numpy
    trasaturiDateDeAntrenament.append(imagineCaNpArray.flatten())  # folosim flateen ca sa facem arrayul sa fie 1D

trasaturiPrelucrateAntrenament = transformaListaInValoriDiscrete(trasaturiDateDeAntrenament, capeteIntervale)

# prelucram datele pt imaginile de validare
trasaturiImaginiDeValidare = []
for i in range(len(numeImaginiDeValidare)):
    numeImagine = numeImaginiDeValidare[i]
    imagine = Image.open(directorCuDate + "val_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)
    trasaturiImaginiDeValidare.append(imagineCaNpArray.flatten())

trasaturiPrelucrateValidare = transformaListaInValoriDiscrete(trasaturiImaginiDeValidare, capeteIntervale)

modelNaiveBayes.fit(trasaturiPrelucrateAntrenament, claseCsvDeAntrenament)
predictiiValidare = modelNaiveBayes.predict(trasaturiPrelucrateValidare)
acuratete = calculeazaAcuratete(predictiiValidare, claseCsvDeValidare)
print("Acuratete de validare pentru " + str(nrIntervale) + " intervale: " + str(acuratete))
dateDespreAcuratete()

#prelucram datele pt imaginile de test
trasaturiImaginiDeTest = []
for i in range(len(numeImaginiDeTest)):
    numeImagine = numeImaginiDeTest[i]
    imagine = Image.open(directorCuDate + "test_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)
    trasaturiImaginiDeTest.append(imagineCaNpArray.flatten())

trasaturiPrelucrateTest = transformaListaInValoriDiscrete(trasaturiImaginiDeTest, capeteIntervale)

predictiiTest = modelNaiveBayes.predict(trasaturiPrelucrateTest)

fisierDePredictii = pd.DataFrame({"Image": CSVdeTest["Image"], "Class": predictiiTest})  #face fisierul unde pune clasa care a fost prezisa pt fiecare imagine
fisierDePredictii.to_csv(directorCuDate + "submission.csv", index=False)



