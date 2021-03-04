from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
path = r"C:\Users\HP\Documents\Py Programs\wine.csv"
# load dataset into Pandas DataFrame
df = pd.read_csv(path)
# Dodaj nazive kolona koje su date
features = ['a','b','c','d','e','f']
# Separating out the features
X = df.loc[1:, features].values
# Separating out the target
#Ddaj vrijednost
y = df.loc[1:,['cojk']].values.ravel()
print(X.shape)
print(y.shape)

#Dvije komponente
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
fig=plt.figure()
finalDf = pd.concat([pd.DataFrame(X_r2,columns = ['1', '2']), df[['cojk']]], axis = 1)
axx=fig.add_subplot(131)
axx.scatter(X_r2[:,0],X_r2[:,1])

ax = fig.add_subplot(1,3,2) 
ax.set_xlabel('LDA 1', fontsize = 15)
ax.set_ylabel('LDA 2', fontsize = 15)
ax.set_title('2 component LDA', fontsize = 20)
#ax.legend(['LDA', 'gas', 'hehe'],loc='best', shadow='false', scatterpoints=1)
targets = ['1', '2', '3']
markers = ['x', '.', 'o']
#Dodaj int za brojeve, ili izbrisi za string
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf['cojk'] == int(target)
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
#print(lda.predict([[6,3,4,1]]))
print('---------------------------------')
#Tacnost modela
print('--------------LDA score---------------')
print(lda.score(X,y))
print('---------------------------------')
#Jedna komponenta
lda = LinearDiscriminantAnalysis(n_components=1)
X_r1 = lda.fit(X, y).transform(X)
####################
gas=np.ones(150)
mmmm=np.dot(5,gas)
dodatna=pd.DataFrame(mmmm,columns=['2'])
finalDf1 = pd.concat([pd.DataFrame(X_r1,columns = ['1']),dodatna,  df[['cojk']]], axis = 1).dropna()
ax1 = fig.add_subplot(1,3,3) 
ax1.set_title('1 component LDA', fontsize = 20)
targets1 = targets
markers = ['x', '.', 'o']

for target, marker in zip(targets1,markers):
    indicesToKeep = finalDf1['cojk'] == int(target)
    ax1.scatter(finalDf1.loc[indicesToKeep, '1'], finalDf1.loc[indicesToKeep, '2']           
               , marker = marker
               , s = 50)
ax1.legend(targets1)
ax1.set_xlim([0,5])
ax1.set_ylim([0,8])
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
#print(lda.predict([[6,3,4,1]]))
print('---------------------------------')
#Tacnost modela
print('--------------LDA Score-------------')
print(lda.score(X,y))
print('---------------------------------')
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()

from sklearn.preprocessing import StandardScaler
x=X
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['cojk']]], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
markers = ['x', '.', 'o']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf['cojk'] == int(target)
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , marker = marker
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
print('Ovjasnjaena varijansa')
print(pca.explained_variance_ratio_)
