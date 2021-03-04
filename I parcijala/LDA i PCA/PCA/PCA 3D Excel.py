import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

path = r"C:\Users\HP\Documents\Py Programs\irisevi.csv"
# load dataset into Pandas DataFrame
df = pd.read_csv(path, names=['sepal length','sepal width','petal length','petal width','target'])
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
centers = [[1, 1], [-1, -1], [1, -1]]

X = df.loc[1:, features].values
print(X.shape)
# Separating out the target
y = df.loc[1:,['target']].values
yy=pd.DataFrame(y, columns=['target'])



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
print(X.shape)
principalComponents=X
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])

finalDf = pd.concat([principalDf, yy], axis = 1)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
markers = ['x', '.', 'o']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , marker = marker
               , s = 50)
ax.legend(targets)
plt.show()