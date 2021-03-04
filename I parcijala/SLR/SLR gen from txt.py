import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
points = np.genfromtxt(r'C:\Users\HP\Documents\Py Programs\SLR\data.csv', delimiter=',')

#Extract columns
X = array(points[:,0]).reshape(-1,1)
Y = array(points[:,1]).reshape(-1,1)
print(X)
lm = LinearRegression()
model = lm.fit(X, Y)
#ispis parametara modela
print('alpha =',model.intercept_)
print('beta =', model.coef_)
#linearna funkcija y_model (regresirana)
y_model=lm.predict(X)
#dijagrami
plt.plot(X,y_model)
plt.scatter(X,Y,c='orange')
plt.title('Linearna regresija y=alfa + beta X')
plt.show()
