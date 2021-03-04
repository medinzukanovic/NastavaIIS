from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA()

X_reduced = pca.fit_transform(X)

rf = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y)
#print(X_test)
#rf.fit(X_train, y_train)
#print(rf.predict_proba(X_test))
#print(rf.score(X_test, y_test))
print(X_test)
print(pca.inverse_transform(X_test))
#print(y_test)
