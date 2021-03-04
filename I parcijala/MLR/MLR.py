import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#######################3
#Import podataka
df=pd.read_csv(r'C:\Users\HP\Documents\Py Programs\IIS rok\MLR_data.csv')
x = df.iloc[:,[0,1]].values
y = df.iloc[:,2].values # slučajna greška
x1=np.array(x[:,0])
x2=np.array(x[:,1])
###################
#Razdvajanje na train i test
test_velicina=0.2  #procentualno 

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_velicina) 
#############################
class MultivariateRegression():
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
    
    def fit(self, x, y):
        x = np.hstack((np.ones((x.shape[0], 1)), x))

        b1 = np.linalg.inv(np.dot(x.T, x))
        b2 = np.dot(x.T, y)
        b = np.dot(b1, b2)
        
        self.intercept_ = b[0]
        self.coef_ = b[1:]
    
    def predict(self, x):
        return np.dot(x, self.coef_) + self.intercept_
#########################333
reg = MultivariateRegression()
reg.fit(x_train, y_train)
########################
print('Intercept:')
print(reg.intercept_)
print('koeficijenti')
print(reg.coef_)
##############################
x1_ = np.linspace(np.min(x_test[:,0]), np.max(x_test[:,0]), 10)
x2_ = np.linspace(np.min(x_test[:,1]), np.max(x_test[:,1]), 10)
x1v, x2v = np.meshgrid(x1_, x2_)
x1v, x2v = x1v.reshape(-1, 1), x2v.reshape(-1,1)

xv = np.hstack((x1v, x2v))
y_pred = reg.predict(xv)
##################################
#####################################
#greške za cijeli model

from sklearn.metrics import r2_score, mean_squared_error
#R^2 (coefficient of determination) regression score function: metrics.r2_score(y_true, y_pred, *[, …])
print('Parametri cijelog modela bez train/test')
R_2=r2_score(np.array(y).reshape(-1,1), reg.predict(x))
print('R-squared=', R_2)
# Mean squared error regression loss: metrics.mean_squared_error(y_true, y_pred, *) 
MSE=mean_squared_error(y, reg.predict(x))
print('MSE=', MSE)
#
print('Parametri testnog dijela modela')
R_22=r2_score(y_test, reg.predict(x_test))
print('Za testni dio R-squared=', R_22)
# Mean squared error regression loss: metrics.mean_squared_error(y_true, y_pred, *) 
MSE2=mean_squared_error(y_test, reg.predict(x_test))
print('Za testni dio MSE=', MSE2)
##################################
predvidjanje=reg.predict([30,5])
print('Predviđanje \n ',predvidjanje)
############################3
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(131, projection='3d')
ax.scatter3D(x1, x2, y)
ax1 = fig.add_subplot(132, projection='3d')
ax1.scatter3D(x_train[:,0], x_train[:,1],y_train)
ax2 = fig.add_subplot(133, projection='3d')
ax2.scatter3D(x_test[:,0], x_test[:,1],y_test)
ax2.plot_trisurf(x1v.ravel(), x2v.ravel(), y_pred.ravel(), alpha=0.3, color='cyan', shade=False)
ax.title.set_text('Cijeli model')
ax1.title.set_text('Train podaci')
ax2.title.set_text('Test podaci sa regresijom')
plt.show()
Be2==1/(1/B0+V0/({1,'Pa'}*((V0+VA)*100000*sqrt(abs(A.p/{1,'Pa'}))))+VA/((V0+VA)*BC));
1/B0+V0/({1,'Pa'}*((V0+VA)*100000*sqrt(abs(A.p/{1,'Pa'}))))+VA/((V0+VA)*BC)