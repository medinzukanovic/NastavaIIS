import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\HP\Documents\Py Programs\irisevi.csv"
# load dataset into Pandas DataFrame
df = pd.read_csv(path, names=['sepal length','sepal width','petal length','petal width','target'],header=None)

print(df)
from sklearn.preprocessing import StandardScaler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[1:, features].values

# Separating out the target
y = df.loc[1:,['target']].values
yy=pd.DataFrame(y, columns=['target'])
print(y)
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, yy], axis = 1)

print(finalDf)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
markers = ['x', '.', 'o']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , marker = marker
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
print('Ovjasnjaena varijansa')
print(pca.explained_variance_ratio_)
