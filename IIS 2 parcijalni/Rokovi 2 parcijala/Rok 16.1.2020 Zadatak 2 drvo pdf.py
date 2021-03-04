from sklearn.datasets import load_iris
from sklearn import tree
import os
import pandas as pd
df=pd.read_csv(r'C:\Users\HP\Documents\Py Programs\FAJLOVI DRUGI PARCIJALNI\Play.csv', sep=";")
labels = ("X1","X2","X3","X4") 
X=df.iloc[:,[1,2,3,4]].values
y=df.iloc[:,5].values
#Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
clf = tree.DecisionTreeClassifier(criterion="gini",splitter="best")
#parametri DTC:
#random_state = 0, criterion = crit,splitter = split,max_depth = depth,min_samples_split=min_split, min_samples_leaf=min_leaf
#crit = ["gini", "entropy"],split = ["best", "random"],depth=[1,2,3,4],min_split=(0.1,1),min_leaf=(0.1,0.5))
clf = clf.fit(X_train, y_train)
import graphviz  
#Spemi ga kao PDF fajl
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=labels,class_names=['0','1'],filled=True,rounded=True,special_characters=True)  
graph = graphviz.Source(dot_data)  
#Promjenuti putanju dobit ce se PDF pod tim imenom, moze i antivirus malo zezat ja moro ugayiti 
graph.render(r'C:\Users\HP\Documents\Py Programs\IIS 2 parcijalni\Rokovi 2 parcijala\iris proba')
#predikcija
print('predvidjanje',clf.predict([[2., 2.,1,2]]))
#tacnost
print('score=',clf.score(X_test,y_test) )