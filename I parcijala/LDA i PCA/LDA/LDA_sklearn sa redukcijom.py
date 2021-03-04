from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names
#Dvije komponente
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
print(X_r2.shape)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)

plt.legend(loc='best', shadow='false', scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
#Model
print('Intercept=',lda.intercept_)
#Koficijenti
print(lda.coef_)
#Objasnjena varijansa
lda.explained_variance_ratio_
print('Prva komponenta =', lda.explained_variance_ratio_[0])
print('Druga komponenta =', lda.explained_variance_ratio_[1])
print('Ukupno =', np.sum(lda.explained_variance_ratio_))
#Predvidjanje
#sepal length = 6, sepal width = 3, petal length = 4, petal width = 1
print(lda.predict([[6,3,4,1]]))
print(target_names[lda.predict([[6,3,4,1]])])
#Tacnost modela
print(lda.score(X,y))
#Jedna komponenta
lda = LinearDiscriminantAnalysis(n_components=1)
X_r1 = lda.fit(X, y).transform(X)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.plot(X_r1[y == i, 0], np.zeros_like(X_r1[y == i]), color=color, alpha=.8, lw=lw, label=target_name)

plt.legend(loc='best', shadow='false', scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
#intercept
print('Intercept=',lda.intercept_)
#koeficijenti
print(lda.coef_)
#objasnjna varijansa
lda.explained_variance_ratio_
print('Prva komponenta =', lda.explained_variance_ratio_[0])
print('Ukupno =', np.sum(lda.explained_variance_ratio_))
#predvidjanje
#sepal length = 6, sepal width = 3, petal length = 4, petal width = 1
print(lda.predict([[6,3,4,1]]))
print(target_names[lda.predict([[6,3,4,1]])])
#Tacnost modela

print(lda.score(X,y))
