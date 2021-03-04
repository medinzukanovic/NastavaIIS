# importovanje biblioteka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
# importovanje dataseta
podaci = pd.read_csv(r'C:\Users\HP\Documents\Py Programs\FAJLOVI DRUGI PARCIJALNI\SVM.csv')
print(podaci) 
X = podaci.iloc[:,[1,3]].values 
print(X)
# ovdje sam samo provjerio y i isprintao
y = podaci.Type
print(y)
# ovim kodom pretvaramo Muffin i Cupcake u 0 i 1 brojne vrijednosti
# i to sam isprinto radi provjere
y=np.where(podaci['Type']=='Muffin',0,1)
print(y)
# djeljenje na test i train podatke. U zadatku je receno train/test = 80%/20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0 # SVM regularization parameter

#koristili smo linearni kernel
svc = svm.SVC(kernel='linear', C=1,gamma='auto').fit(X_train, y_train)
#mozemo jeo korititi: 
#rbf kernel
#svc = svm.SVC(kernel='rbf', C=1,gamma='auto').fit(X, y)
#create a mesh to plot in
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#plotanje dijagrama (slike)
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
plt.xlabel('Flour')
plt.ylabel('Sugar')
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

#predvidjanje
def muffin_or_cupcake (Flour,Sugar):
    if(svc.predict([[Flour,Sugar]]))==0:
        print('Rijec je o Muffinu')
    else:
        print('Rijec je o Cupcakeu')
muffin_or_cupcake(40,20)
# samo u liniji koda iznad mjenjamo vrijednosti i dobit ćemo ispis rezultata


# drugi način kako uraditi predviđanje
p2=[[40,20]]
pred = svc.predict(p2)
print(pred)

# vise o ovome na ovom linku: https://www.youtube.com/watch?v=N1vOgolbjSc&feature=youtu.be