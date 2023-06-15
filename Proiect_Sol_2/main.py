import pandas as pd
import numpy as np
from pandas import read_csv
from PIL import Image

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


def classificaOSinguraImagine(tipDistanta, numarVecini, imagineCareTrabuieTestata):
    if tipDistanta == 'Euclidian':
        valoareDistanta = np.sqrt(np.sum(np.square(trasaturiDateDeAntrenament - imagineCareTrabuieTestata), axis=(1, 2, 3)))  # distanta intre 2 imagini; facem pe 3 axe pt ca imaginea e de forma nrImagini x nrPixeliPeRand x nrPixeliPeColoana x 3(RGB)
    elif tipDistanta == 'Manhattan':
        valoareDistanta = np.sum(np.abs(trasaturiDateDeAntrenament - imagineCareTrabuieTestata), axis=(1, 2, 3))

    indiciSortati = np.argsort(valoareDistanta)  # indicii celor mai apropiati vecini sunt primii
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


def clasificaImagini(tipDistanta, numarVecini, trasaturiImagini):
    numarImagini = len(trasaturiImagini)  # trasaturiImagini e de forma nrImagini x nrPixeliPeRand x nrPixeliPeColoana x 3(RGB)
    clasePrezise = []

    for i in range(numarImagini):
        clasaPrezisa = classificaOSinguraImagine(tipDistanta, numarVecini, trasaturiImagini[i])
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


def construiesteMatriceaDeConfuzie():   #se face pentru datele de validare
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


##################### MAAAIIIIIINNNNN #############

deschiderInitialaDeFisiere()

trasaturiDateDeAntrenament = []
for i in range(len(numeImaginiDeAntrenament)):
    numeImagine = numeImaginiDeAntrenament[i]
    imagine = Image.open(directorCuDate + "train_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)  # vrem sa avem datele legate de pixeli sub forma unui array din numpy
    trasaturiDateDeAntrenament.append(imagineCaNpArray)

trasaturiDateDeValidare = []
for i in range(len(numeImaginiDeValidare)):
    numeImagine = numeImaginiDeValidare[i]
    imagine = Image.open(directorCuDate + "val_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)  # vrem sa avem datele legate de pixeli sub forma unui array din numpy
    trasaturiDateDeValidare.append(imagineCaNpArray)

trasaturiDateDeTest = []
for i in range(len(numeImaginiDeTest)):
    numeImagine = numeImaginiDeTest[i]
    imagine = Image.open(directorCuDate + "test_images/" + numeImagine)
    imagineCaNpArray = np.array(imagine)  # vrem sa avem datele legate de pixeli sub forma unui array din numpy
    trasaturiDateDeTest.append(imagineCaNpArray)

predictiiValidare = clasificaImagini('Euclidian', 498, trasaturiDateDeValidare)
acuratete = calculeazaAcuratete(predictiiValidare, claseCsvDeValidare)

print("Acuratete de validare: " + str(acuratete))
construiesteMatriceaDeConfuzie()

predictiiTest = clasificaImagini('Euclidian', 81, trasaturiDateDeTest)
fisierDePredictii = pd.DataFrame({"Image": CSVdeTest["Image"], "Class": predictiiTest})  #face fisierul unde pune clasa care a fost prezisa pt fiecare imagine
fisierDePredictii.to_csv(directorCuDate + "submission.csv", index=False)