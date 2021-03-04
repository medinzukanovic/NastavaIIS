# importovanje biblioteka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
# importovanje dataseta
podaci = pd.read_csv(r'C:\Users\HP\Documents\Py Programs\podaci\SVM.csv')
X = podaci.iloc[:,[0,1]].values 
y = podaci.y
plt.scatter(X[:,0],X[:,1])
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=150)
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=1,gamma='auto').fit(X_train, y_train)
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
h = np.abs((x_max / x_min)/100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#plotanje dijagrama (slike)
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(xx.min(), xx.max())
plt.title('SVC sa linear kernel')
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
# drugi način kako uraditi predviđanje
p2=[[-10,-7],[0,7]]
pred = svc.predict(p2)
print(pred)