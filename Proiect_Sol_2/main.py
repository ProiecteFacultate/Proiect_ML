import pandas as pd
import numpy as np
from pandas import read_csv
from PIL import Image
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


def normalizeazaImagine(imagineCaNpArray):
    medieImagine = np.mean(imagineCaNpArray)    #media pixelilor
    deviatieStandard = np.std(imagineCaNpArray)
    imagineNormalizata = (imagineCaNpArray - medieImagine) / deviatieStandard
    return imagineNormalizata


def classificaOSinguraImagine(numarVecini, imagineCareTrabuieTestata):
    diferenta = trasaturiDateDeAntrenament - imagineCareTrabuieTestata
    distanta = np.sqrt(np.sum(np.square(diferenta), axis=(1, 2, 3)))  #Euclidian; distanta intre 2 imagini; facem pe 3 axe pt ca imaginea e de forma 12000 x 64 x 64 x 3
  #  distanta = np.sum(np.abs(trasaturiDateDeAntrenament - imagineCareTrabuieTestata), axis=(1, 2, 3))  #Manhattan

    indiciSortati = np.argsort(distanta)  # indicii celor mai apropiati vecini sunt primii
    aparitiiClase = [0 for x in range(96)]

    for i in range(numarVecini):
        indice = indiciSortati[i]
        clasa = claseCsvDeAntrenament[indice]
        aparitiiClase[int(clasa)] += 1

    clasaCuNrMaxDeAparitii = -1
    numarMaximAparitii = -1
    for i in range(96):
        if aparitiiClase[i] > numarMaximAparitii:
            numarMaximAparitii = aparitiiClase[i]
            clasaCuNrMaxDeAparitii = i

    return clasaCuNrMaxDeAparitii


def clasificaImagini(numarVecini, trasaturiImagini):
    numarImagini = len(trasaturiImagini)  # trasaturiImagini e de forma nrImagini x nrPixeliPeRand x nrPixeliPeColoana x 3(RGB)
    clasePrezise = []

    for i in range(numarImagini):
        clasaPrezisa = classificaOSinguraImagine(numarVecini, trasaturiImagini[i])
        clasePrezise.append(clasaPrezisa)

    return clasePrezise


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

        if truePositive + falsePositive != 0:   #daca nu facem verificarea se poate primi division by zero
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

#     deseneazaMatrice(matriceConfuzie)
#
# def deseneazaMatrice(matrice):
#     plt.rcParams["figure.figsize"] = [50, 50]
#     plt.rcParams["figure.autolayout"] = True
#     fig, ax = plt.subplots()
#     ax.matshow(matrice, cmap='binary')
#     plt.show()

##################### MAAAIIIIIINNNNN #############

deschiderInitialaDeFisiere()

trasaturiDateDeAntrenament = []
for i in range(len(numeImaginiDeAntrenament)):
    numeImagine = numeImaginiDeAntrenament[i]
    imagine = Image.open(directorCuDate + "train_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)  # vrem sa avem datele legate de pixeli sub forma unui array din numpy
    imagineNormalizata = normalizeazaImagine(imagineCaNpArray)
    trasaturiDateDeAntrenament.append(imagineNormalizata)

trasaturiDateDeValidare = []
for i in range(len(numeImaginiDeValidare)):
    numeImagine = numeImaginiDeValidare[i]
    imagine = Image.open(directorCuDate + "val_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)  # vrem sa avem datele legate de pixeli sub forma unui array din numpy
    imagineNormalizata = normalizeazaImagine(imagineCaNpArray)
    trasaturiDateDeValidare.append(imagineNormalizata)

trasaturiDateDeTest = []
for i in range(len(numeImaginiDeTest)):
    numeImagine = numeImaginiDeTest[i]
    imagine = Image.open(directorCuDate + "test_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)  # vrem sa avem datele legate de pixeli sub forma unui array din numpy
    imagineNormalizata = normalizeazaImagine(imagineCaNpArray)
    trasaturiDateDeTest.append(imagineNormalizata)

numarVecini = 26
predictiiValidare = clasificaImagini(numarVecini, trasaturiDateDeValidare)
acuratete = calculeazaAcuratete(predictiiValidare, claseCsvDeValidare)
print("Acuratete de validare folosind ditanta Manhattan pentru " + str(numarVecini) + ": " + str(acuratete))
dateDespreAcuratete()

predictiiTest = clasificaImagini(numarVecini, trasaturiDateDeTest)
fisierDePredictii = pd.DataFrame({"Image": CSVdeTest["Image"], "Class": predictiiTest})  #face fisierul unde pune clasa care a fost prezisa pt fiecare imagine
fisierDePredictii.to_csv(directorCuDate + "submission.csv", index=False)