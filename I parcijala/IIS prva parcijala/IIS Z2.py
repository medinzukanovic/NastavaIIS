from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
path = r"C:\Users\HP\Documents\Py Programs\FAJLOVI ISPIT IIS\LDA.csv"
df = pd.read_csv(path)
features = ['age','weight_kg','height_cm','bmd']
X = df.loc[:, features].values
y = df.loc[:,['medication']].values.ravel()
####33
test_velicina=0.2 # 20% test
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = test_velicina, random_state=44) 
###
#Dvije komponente
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(xtrain, ytrain).transform(xtrain)
finalDf = pd.concat([pd.DataFrame(X_r2,columns = ['1', '2']), df[['medication']]], axis = 1)
print(finalDf)
########
#Model
print('---------------------Model s dvije komponente-------------------')
print('Intercept=',lda.intercept_)
print('Koeficijent:')
print(lda.coef_)
print('---------------------------------')
############
print('-------Objasnjena varijansa-------------')
lda.explained_variance_ratio_
print('Prva komponenta =', lda.explained_variance_ratio_[0])
print('Druga komponenta =', lda.explained_variance_ratio_[1])
print('Ukupno =', np.sum(lda.explained_variance_ratio_))
print('---------------------------------')

######3
fig=plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('LDA 1', fontsize = 15)
ax.set_ylabel('LDA 2', fontsize = 15)
ax.set_title('2 component LDA', fontsize = 20)
ax.legend(['LDA', 'gas', 'hehe'],loc='best', shadow='false', scatterpoints=1)
targets = ['0', '1', '2']
markers = ['x', '.', 'o']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf['medication'] == int(target)
    ax.scatter(finalDf.loc[indicesToKeep, '1'], finalDf.loc[indicesToKeep, '2'], marker = marker, s = 50)
ax.legend(targets)
plt.show()
#################


#Tacnost modela
print('--------------LDA score---------------')
print(lda.score(xtest,ytest))
print('---------------------------------')
