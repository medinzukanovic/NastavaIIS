import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score

podaci = pd.read_csv(r'C:\Users\HP\Documents\Py Programs\I parcijala\FAJLOVI ISPIT IIS\LR.csv')
X = podaci.iloc[:,[2,3,4]].values 
y = podaci.iloc[:,1].values

# djeljenje na test i train podatke. U zadatku je receno train/test = 80%/20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#za dobru regsiju prema preporuci iz literature dobro je normalizirat ulazne podatke
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
#Ovako pozivas neuralnu mrežu
model = Sequential()
#########################################U zavisnosti koji je problem
#Prvi sloj ulaz za regreyiju kolko ima iksova
model.add(Dense(20, input_dim=3, activation='tanh'))
model.add(Dense(10, activation='tanh'))
#izlazni lejer ako je regresije 1
model.add(Dense(1, activation='linear'))
###########################################
#FITOVANJE
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','mse','mape'])
model.fit(normalized_X,y_train, epochs=300, batch_size=100, verbose=1)
#Opis modela
print(model.summary())
#tačnost
loss,accuracy,mse,mape=model.evaluate(X_test, y_test)
print('loss:', loss)
print('accuracy:',accuracy)
print('mse:', mse)
print('mape', mape)
#Predidjanje promjeni tačku dadne prof
a= np.array([[60,0.08,0.1]])
print(model.predict(a))
