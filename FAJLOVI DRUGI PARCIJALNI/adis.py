import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
podaci = pd.read_csv('NN_reg.csv')
X = podaci.iloc[:,[0,1]].values 
y = podaci.iloc[:,2].values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],y)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=120)
model = Sequential()
model.add(Dense(20, input_dim=2, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mape'])
model.fit(X_train,y_train, epochs=50000,verbose=0)
predikcija=model.predict(X)
print(X.shape)
print(predikcija.shape)
fig1=plt.figure()
ax1=fig1.add_subplot(111, projection='3d')
ax1.scatter(X[:,0],X[:,1],y,c='red',label='stvarno')
ax1.scatter(X[:,0],X[:,1],predikcija, c='green',label='predvidjanje')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('Y')
ax1.legend()
plt.show()
loss,mse,mape=model.evaluate(X_test, y_test)
print('loss:', loss)
print('mse:', mse)
print('mape', mape)
a= np.array([[0,0]])
b=np.array([[-0.5,0.5]])
print('Predviđanje 1')
print(model.predict(a))
print('Predviđanje 2')
print(model.predict(b))
