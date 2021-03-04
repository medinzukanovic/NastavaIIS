#KNN KLASTERING
#IMPORTOVNAJE BIBILOTEKA
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#IMPORT PODATAKA
from sklearn.datasets import load_iris
iris = load_iris()
#PRIKAZ PODATAKA
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['class'] = iris.target
df['class'] = df['class'].map({0:iris.target_names[0], 1:iris.target_names[1], 2:iris.target_names[2]})
#OSOBINE DF
print('***********')
print(df.head(10))
print('***********')
print(df.describe())
print('***********')
#DODAVANJE X i Y
x = iris.data
y = iris.target.reshape(-1, 1) #column vector
#y = iris.target.reshape(1,-1) #row vector
print(x.shape, y.shape)
print('***********')
#TRAIN TEST
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
print(x_train.shape, y_train.shape)
print('***********')
print(x_test.shape, y_test.shape)
print('***********')
#FITOVANJE
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2)
knn.fit(x_train, y_train.ravel())
#PRIKAZ REZULTATA
list_res = []
for p in [1, 2]:
    knn.p = p
    
    for k in range(1, 10, 2):
        knn.n_neighbors = k
        y_pred = knn.predict(x_test)
        acc = accuracy_score(y_test, y_pred)*100
        list_res.append([k, 'l1_distance' if p == 1 else 'l2_distance', acc])
        
df1 = pd.DataFrame(list_res, columns=['k', 'dist. func.', 'accuracy'])
print(df1)
print('***********')