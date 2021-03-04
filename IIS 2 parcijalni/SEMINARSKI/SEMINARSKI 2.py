##########################################################################################################
#IMPORT BIBLIOTEKA
import tensorflow as tf
import keras
import numpy as np
import glob
import numpy as np
##########################################################################################################
#IMPORT SLIKA
folder=r"C:\Users\HP\Desktop\SLIKE ZA SEMINARSKI"
files = glob.glob (r"C:\Users\HP\Desktop\SLIKE ZA SEMINARSKI/*.JPG")
##########################################################################################################
#OBRADA SLIKA
from keras.preprocessing.image import img_to_array, load_img
train_files = []
Y_train = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#KONVERZIJA U "categorical"  ZATO ŠTO POSTOJE 2 DISKRETNE KLASE
Y_train=tf.keras.utils.to_categorical(Y_train, num_classes=None, dtype="float32")
print('Y_train keras shape')
print(Y_train.shape)
i=0
for _file in files:
    train_files.append(_file)
print('**************************')    
print("Ukupno slika: %d" % len(train_files))
print('**************************')
##########################################################################################################
# ORGINALNE DIMENZIJE
sirina_slike = 1200
visina_slike = 1600
##########################################################################################################
#REDUKOVANE DIMENZIJA
omjer = 4
sirina_slike = int(sirina_slike / omjer)
visina_slike = int(visina_slike / omjer)
##########################################################################################################
#BROJ KANALA (RGB-U boji)
broj_kanala_boja = 3
##########################################################################################################
#KREIRANJE DATASETA
print('**************************')
dataset = np.ndarray(shape=(len(train_files), visina_slike, sirina_slike, broj_kanala_boja),dtype=np.float32)
print('Dataset SHAPE')
print(dataset.shape)
print('**************************')
#ITERATIVNO UBACIVANJE SLIKA U DATASET TJ. FORMIRANJE MATRICA
i = 0
for _file in train_files:
    
    img = load_img(_file)  # this is a PIL image
    img.thumbnail((sirina_slike, visina_slike))
    # Convert to Numpy Array
    x = img_to_array(img)  
    x = x.reshape(visina_slike, sirina_slike, broj_kanala_boja)
    # Normalize
    x = (x - 128.0) / 128.0
    dataset[i] = x
    i += 1
    if i % 2 == 0:
        print("%d SLIKA DODANA" % i)
print("SVE SLIKE DODANE!")
##########################################################################################################
#RAZDVAJANJE TRAIN TEST
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataset, Y_train, test_size=0.2, random_state=33)
print("Veličina TRAINA: {0}, Veličina TESTA: {1}".format(len(X_train), len(X_test)))
##########################################################################################################
#ULAZNI PODACI ZA CNN
num_classes = 2 # 2 klase zato što imamo 2 oblika
#KONVERZIJA U NUMERIČKE VRIJEDNSOTI
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#SKALIRANJE PODATAKA
X_train /= 255
X_test /= 255
#ISPIS PARAMETARA
print('**************************')
#print('X_train:')
#print(X_train)
#print('Y_train')
#print(Y_train)
print('X_train.shape:')
print(X_train.shape)
print('Y_train.shape:')
print(Y_train.shape)
#print(Y_train.shape)
print('**************************')
##########################################################################################################
#KONVOLUCIJSKA NEURALNA MREŽA
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten 
from keras.layers import MaxPooling2D, Dropout
#SEKVENCIJALNI MODEL, MODELIRA SE LAYER LAYER
model = Sequential()
#1. LAYER Konvolucijski, dodaje filter na podatke, ulazni podaci moraju biti 4D, akreira 32 filtera i preko njih analizira [3x3] podataka u matrici ulaznoj
#ReLu aktivacija podrazumijeva linearnu funkciju koja inpute prebacuje direktno u outpute ako pozitivni, a ako su negativni dodaje im vrijednost 0
model.add(Conv2D(32, kernel_size=(3,3),
                     activation='relu',input_shape=(400,300,3)))
#2. LAYER MaxPooling rješava male varijacije, obično se koristi iza konvolucije koja ima ReLu aktivaciju, postoji Max i Average Pooling
model.add(MaxPooling2D(pool_size=(3, 3)))
#3. LAYER Konvolucijski, dodaje filter na podatke, ulazni podaci moraju biti 4D, akreira 64 filtera i preko njih analizira [3x3] podataka u matrici ulaznoj
model.add(Conv2D(64, (3, 3), activation='relu'))
#4. LAYER MaxPooling rješava male varijacije, obično se koristi iza nelinearne konvolucije relu, postoji Max i Average Pooling
model.add(MaxPooling2D(pool_size=(3, 3)))
#5. LAYER Flattern za poravnanje podataka, da se dobije 1D output
model.add(Flatten())
#6. LAYER Svakom neuronu (1024 neurona) se pripiše svaki podatak iz prethodnog layera
model.add(Dense(1024, activation='relu'))
#7. LAYER Dropout da se odbaci pola podataka
model.add(Dropout(0.5))
#8. LAYER Svakom neuronu (2 neurona zbog dvije klase objekara) se pripiše svaki podatak iz prethodnog layera
#Softmax aktivacijska funkcija koja pretvara izlaz u nlaži oblik argmax funkcije dodavajući vjerovatnoće svakom elementu prema najvećem elementu matrice
model.add(Dense(num_classes, activation='softmax'))
#KOMPAJLIRANJE MODELA
#adam optimizer efikasan optimizacijski algoritam. Razlikuje se od gradient descent algoritma po tome što ima promjenjiv koeficijent učenja alfa.
#metrics accuracy iterqativno poredi predikcije i stvarne podatke
#loss categorical_crossenreopy računa unaksnu entropiju tj pad entropije za dvije koristene vjerovatnoce: optimiziranu i stvarnu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#FITOVANJE TRAIN PODATAKA
#Epochs podrazumijeva broj prolaza kompletnog trening dataseta kroz neuralnu mrežu
X_train=X_train.reshape(-1,400,300,3)
model.fit(X_train, Y_train, epochs=20)
#EVALUACIJA MODELA
score = model.evaluate(X_test.reshape(-1,400,300,3), Y_test, verbose=1)
#ISPIS PARAMETARA
print('**************************')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('**************************')
##########################################################################################################
#PREDIKCIJA
print('**************************')
print('PREDVIĐANJE')
#print(model.predict_classes(X_test[:2].reshape(-1,300,400,3)))
print(np.argmax(model.predict(X_test.reshape(-1,400,300,3)), axis=-1))
print('**************************')
print('STVARNI PODACI')
print(np.argmax(Y_test[:6],axis=-1))
print('**************************')
print(model.summary())
##########################################################################################################