# importovanje biblioteka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
# importovanje dataseta
# za atribute uzeti variance, skewness, curtosis i entropy
podaci = pd.read_csv(r'C:\Users\HP\Documents\Py Programs\IIS 2 parcijalni\Rokovi septembarski\bank_notes.csv')
X = podaci.iloc[:,[0,1,2,3]].values 

#print('X=', X)
# ovdje sam samo provjerio y i isprintao
y = podaci.Target
plt.show()
# djeljenje na test i train podatke. U zadatku je receno train/test = 80%/20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#koristili smo linearni kernel
svc = svm.SVC(kernel='linear', C=1,gamma='auto').fit(X_train, y_train)
Z = svc.predict(X_train)
#X1X2 graf
fig=plt.figure()
fig.suptitle('SVC sa linear kernel')
ax1=plt.subplot(3, 3, 1)
ax1.scatter(X_train[:, 0], X_train[:, 1], c=Z)
ax1.set(xlabel='x1', ylabel='x2')

#X1X3 graf
ax2=plt.subplot(3, 3, 2)
ax2.scatter(X_train[:, 0], X_train[:, 2], c=Z)
ax2.set(xlabel='x1', ylabel='x3')

#X1X4 graf
ax3=plt.subplot(3, 3, 3)
ax3.scatter(X_train[:, 0], X_train[:, 3], c=Z)
ax3.set(xlabel='x1', ylabel='x4')

#X2X3 graf
ax4=plt.subplot(3, 3, 5)
ax4.scatter(X_train[:, 1], X_train[:, 2], c=Z)
ax4.set(xlabel='x2', ylabel='x3')

#X2X4 graf
ax5=plt.subplot(3, 3, 6)
ax5.scatter(X_train[:, 1], X_train[:, 3], c=Z)
ax5.set(xlabel='x2', ylabel='x4')

#X3X4 graf
ax6=plt.subplot(3, 3, 9)
ax6.scatter(X_train[:, 2], X_train[:, 3], c=Z)
ax6.set(xlabel='x3', ylabel='x4')

plt.show()

# Dio koda koji daje support vektore
print('Support vektori=',svc.support_vectors_)

# Prikazivanje indexa tačaka koje prestavljaju support vektore
print('Indexa tačaka koje prestavljaju support vektores=',svc.support_)

# Broj support vektora za svaku klasu
print('Broj support vektora za svaku klasu=',svc.n_support_)
# Tačnost modela
from sklearn.metrics import accuracy_score
y_pred = svc.predict(X_test)
print('Tačnost modela=',accuracy_score(y_test,y_pred))

#predvidjanje
def predikcija (x1,x2,x3,x4):
    if(svc.predict([[x1,x2,x3,x4]]))==0:
        print('Predviđanje je:')
        print('0')
    else:
        print('Predviđanje je:')
        print('1')

#unijet brojeve
predikcija(0.74067, 1.7299, -3.1963, -0.1457)
