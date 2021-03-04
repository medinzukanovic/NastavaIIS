#Linearna regresija
#Importovanje biblioteka
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
##################
#importovanje fajla
dataset = pd.read_csv(r'C:\Users\HP\Documents\Py Programs\ROK 12.12.2018\student_scores.csv')
########################
# input nezavisna varijabla
#prvo red pa kolona

x = (dataset.iloc[:, 0].values)
# output zavisna varijabla
y = dataset.iloc[:, 1].values
##############################
###################
#Razdvajanje na train i test
test_velicina=0.1

from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = test_velicina) 
#################################
x_train=np.array(xtrain).reshape(-1,1)
y_train= np.array(ytrain).reshape(-1,1)
x_test=np.array(xtest).reshape(-1,1)
y_test=np.array(ytest).reshape(-1,1)
lm = LinearRegression()
model_train = lm.fit(x_train, y_train)
#ispis parametara modela
###############################
print('alpha odsjecak =',model_train.intercept_)
print('beta koeficijent =', model_train.coef_)
#######################################
#linearna funkcija y_test (regresirana)
y_predict=(lm.predict(x_test))
#####################################
#greške za cijeli model

from sklearn.metrics import r2_score, mean_squared_error
#R^2 (coefficient of determination) regression score function: metrics.r2_score(y_true, y_pred, *[, …])
print('Parametri cijelog modela bez train/test')
R_2=r2_score(np.array(y).reshape(-1,1), lm.predict(np.array(x).reshape(-1,1)))
print('R-squared=', R_2)
# Mean squared error regression loss: metrics.mean_squared_error(y_true, y_pred, *) 
MSE=mean_squared_error(np.array(y).reshape(-1,1), lm.predict(np.array(x).reshape(-1,1)))
print('MSE=', MSE)
#
print('Parametri testnog dijela modela')
R_22=r2_score(y_test, y_predict)
print('Za testni dio R-squared=', R_22)
# Mean squared error regression loss: metrics.mean_squared_error(y_true, y_pred, *) 
MSE2=mean_squared_error(y_test, y_predict)
print('Za testni dio MSE=', MSE2)
########################################
#dijagrami
plt.plot(x_test,y_predict)
plt.scatter(x_train,y_train,c='orange')
plt.scatter(x_test,y_test,c='green',marker='x')
plt.title('Linearna regresija y=alfa + beta X')
plt.legend(['Regresija','Train','Test'],loc='best')
######### PREDVIĐANJE ######333
predd=lm.predict(np.array([6]).reshape(-1,1))
print('Za željeni X, y=')
print(predd)
plt.show()