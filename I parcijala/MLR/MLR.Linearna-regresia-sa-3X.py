import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
####################### UNESI PODATKE #######################
#Import podataka
df=pd.read_csv(r'C:\Users\HP\Documents\Py Programs\IIS rok\MLR_data.csv')
x = df.iloc[:,[0,1,2]].values
y = df.iloc[:,4].values
x1=np.array(x[:,0])
x2=np.array(x[:,1])
x3=np.array(x[:,2])

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
reg.fit(x,y)
########################
print('Intercept:')
print(reg.intercept_)
print('koeficijenti')
print(reg.coef_)

########################    UNESI PREVIĐANJE  ############################
nezavisne_varijable=np.array([29,47,14])
predvidjanje=reg.predict(nezavisne_varijable)
print('Predviđanje za sljedeće vrijednosti X: \n', nezavisne_varijable)
print('Predviđanje \n ',predvidjanje)
############################3
fig = plt.figure(figsize=(20, 10))
plt.suptitle('Prvo označava horizontalnu, drugo vetikalnu osu')
plt.subplots_adjust(hspace =0.52)

ax1 = fig.add_subplot(441)
ax1.scatter(x1,y)
ax1.scatter(nezavisne_varijable[0],predvidjanje,marker='x')
ax2 = fig.add_subplot(442)
ax2.scatter(x2,y)
ax2.scatter(nezavisne_varijable[1],predvidjanje,marker='x')
ax3 = fig.add_subplot(443)
ax3.scatter(x3,y)
ax3.scatter(nezavisne_varijable[2],predvidjanje,marker='x')
ax5 = fig.add_subplot(446)
ax5.scatter(x1,x2)
ax5.scatter(nezavisne_varijable[0],nezavisne_varijable[1],marker='x')
ax6 = fig.add_subplot(447)
ax6.scatter(x1,x3)
ax6.scatter(nezavisne_varijable[0],nezavisne_varijable[2],marker='x')
ax8 = fig.add_subplot(4,4,11)
ax8.scatter(x3,x2)
ax8.scatter(nezavisne_varijable[2],nezavisne_varijable[1],marker='x')


ax1.title.set_text('Zavisnost x1 y')
ax2.title.set_text('Zavisnost x2 y')
ax3.title.set_text('Zavisnost x3 y')
ax5.title.set_text('Zavisnost x1 x2')
ax6.title.set_text('Zavisnost x1 x3')
ax8.title.set_text('Zavisnost x3 x2')
plt.show()

######GRESKA AKO BUDE TREBALAAAAA PREKO SKLEARN
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x, y)

print('Intercept=',model.intercept_)
print('Coeff=',model.coef_)

from sklearn.metrics import r2_score, mean_squared_error
#R^2 (coefficient of determination) regression score function: metrics.r2_score(y_true, y_pred, *[, …])
R_2=r2_score(y,model.predict(x))
print('R-squared=', R_2)


# Mean squared error regression loss: metrics.mean_squared_error(y_true, y_pred, *) 
MSE=mean_squared_error(y,model.predict(x))
print('MSE=', MSE)