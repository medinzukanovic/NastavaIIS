from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
path = r"C:\Users\HP\Documents\Py Programs\irisevi.csv"
# load dataset into Pandas DataFrame
df = pd.read_csv(path, names=['sepal length','sepal width','petal length','petal width','target'])


features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
X = df.loc[1:, features].values
# Separating out the target
y = df.loc[1:,['target']].values.ravel()

#Dvije komponente
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
fig=plt.figure()
finalDf = pd.concat([pd.DataFrame(X_r2,columns = ['1', '2']), df[['target']]], axis = 1)



ax = fig.add_subplot(1,2,1) 
ax.set_xlabel('LDA 1', fontsize = 15)
ax.set_ylabel('LDA 2', fontsize = 15)
ax.set_title('2 component LDA', fontsize = 20)
ax.legend(['LDA', 'gas', 'hehe'],loc='best', shadow='false', scatterpoints=1)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
markers = ['x', '.', 'o']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, '1']
               , finalDf.loc[indicesToKeep, '2']
               , marker = marker
               , s = 50)
ax.legend(targets)
#################
#Model
print('---------------------Model s dvije komponente-------------------')
print('Intercept=',lda.intercept_)
#Koficijenti
print(lda.coef_)
print('---------------------------------')
#Objasnjena varijansa
print('-------Objasnjena varijansa-------------')
lda.explained_variance_ratio_
print('Prva komponenta =', lda.explained_variance_ratio_[0])
print('Druga komponenta =', lda.explained_variance_ratio_[1])
print('Ukupno =', np.sum(lda.explained_variance_ratio_))
print('---------------------------------')
#Predvidjanje
#sepal length = 6, sepal width = 3, petal length = 4, petal width = 1
print('Predviđanje:')
print(lda.predict([[6,3,4,1]]))
print('---------------------------------')
#Tacnost modela
print('--------------LDA score---------------')
print(lda.score(X,y))
print('---------------------------------')
#Jedna komponenta
lda = LinearDiscriminantAnalysis(n_components=1)
X_r1 = lda.fit(X, y).transform(X)
####################
dodatna=pd.DataFrame(np.zeros(150),columns=['2'])
finalDf1 = pd.concat([pd.DataFrame(X_r1,columns = ['1']),dodatna,  df[['target']]], axis = 1).dropna()
ax1 = fig.add_subplot(1,2,2) 
ax1.set_title('1 component LDA', fontsize = 20)
targets1 = targets
markers = ['x', '.', 'o']

for target, marker in zip(targets1,markers):
    indicesToKeep = finalDf1['target'] == target
    ax1.scatter(finalDf1.loc[indicesToKeep, '1'], finalDf1.loc[indicesToKeep, '2']           
               , marker = marker
               , s = 50)
ax1.legend(targets1)

#################
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('---------------------Model s jednom komponente-------------------')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#intercept
print('Intercept=',lda.intercept_)
print('---------------------------------')
#koeficijenti
print('LDA koeficijenti')
print(lda.coef_)
print('---------------------------------')
#objasnjna varijansa
print('-------Objasnjena varijansa-------------')
lda.explained_variance_ratio_
print('Prva komponenta =', lda.explained_variance_ratio_[0])
print('Ukupno =', np.sum(lda.explained_variance_ratio_))
print('---------------------------------')
#predvidjanje
#sepal length = 6, sepal width = 3, petal length = 4, petal width = 1
print('--------Predviđanje-----------')
print(lda.predict([[6,3,4,1]]))
print('---------------------------------')
#Tacnost modela
print('--------------LDA Score-------------')
print(lda.score(X,y))
print('---------------------------------')
plt.show()
