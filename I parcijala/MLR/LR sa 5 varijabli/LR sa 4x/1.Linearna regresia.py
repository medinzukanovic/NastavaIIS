import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
####################### UNESI PODATKE #######################
#Import podataka
df=pd.read_csv(r'C:\Users\HP\Documents\Py Programs\FAJLOVI ISPIT IIS\LR.csv')
x = df.iloc[:,[2,3,4,5,6]].values
y = df.iloc[:,1].values
####################
test_velicina=0.2 # 20% test

from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = test_velicina, random_state=44) 
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
#########################
reg = MultivariateRegression()
reg.fit(xtrain,ytrain)
########################
print('Intercept:')
print(reg.intercept_)
print('koeficijenti')
print(reg.coef_)

fig = plt.figure(figsize=(20, 10))
plt.suptitle('Prvo označava horizontalnu, drugo vetikalnu osu')
plt.subplots_adjust(hspace =0.52)
ax1 = fig.add_subplot(151)
ax1.scatter(xtest[:,0],ytest)
ax2 = fig.add_subplot(152)
ax2.scatter(xtest[:,1],ytest)
ax3 = fig.add_subplot(153)
ax3.scatter(xtest[:,2],ytest)
ax4 = fig.add_subplot(154)
ax4.scatter(xtest[:,3],ytest)
ax5=fig.add_subplot(155)
ax5.scatter(xtest[:,4],ytest)
ax1.title.set_text('Zavisnost x1 y')
ax2.title.set_text('Zavisnost x2 y')
ax3.title.set_text('Zavisnost x3 y')
ax4.title.set_text('Zavisnost x4 y')
ax5.title.set_text('Zavisnost x5 y')
plt.show()
from sklearn.metrics import r2_score, mean_squared_error
MSE=mean_squared_error(ytest,reg.predict(xtest))
print('MSE=', MSE)
print('Parametri testnog dijela modela')
R_22=r2_score(ytest, reg.predict(xtest))
print('Za testni dio R-squared=', R_22)