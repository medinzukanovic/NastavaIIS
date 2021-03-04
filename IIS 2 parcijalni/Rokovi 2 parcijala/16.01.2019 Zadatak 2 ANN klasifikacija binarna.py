import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
#import NAPOMENA: nađen primjer sa 4 varijable
df = pd.read_csv(r'C:\Users\HP\Documents\Py Programs\IIS 2 parcijalni\Rokovi 2 parcijala\molecular_activity.csv')
x = df.loc[:,['prop_1','prop_2','prop_3','prop_4']]
x = np.asarray(x).astype(np.float32)
y = df.loc[:,['Activity']]
y = np.asarray(y).astype(np.float32)
#Train Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print("Veličina TRAINA: {0}, Veličina TESTA: {1}".format(len(X_train), len(X_test)))
#Kreiranje neuronske mreže
model = Sequential()
model.add(Dense(32, input_dim=4, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1,activation='softmax'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','mse','mape'])
#moze biti i metrics=['accuracy']
print(model.summary())
model.fit(X_train, Y_train, epochs=1000, verbose=0)
#tačnost
loss,accuracy,mse,mape=model.evaluate(X_test, Y_test)
print('loss:', loss)
print('accuracy:',accuracy)
print('mse:', mse)
print('mape', mape)
#predikcija za (-0.78289, 11.3603, -0.37644, -7.0495) i (2.99,60.30,57.46,6.06)
a= np.array([[-0.78289, 11.3603, -0.37644, -7.0495],[2.99,60.30,57.46,6.06]])
print(model.predict(a))