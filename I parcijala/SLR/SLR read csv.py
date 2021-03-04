import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
#importovanje podataka bez headera
df=pd.read_csv(r'C:\Users\HP\Documents\Py Programs\Podaci.csv')
print(df)
X=np.array(df.iloc[:,0]).reshape(-1,1)
Y=np.array(df.iloc[:,1]).reshape(-1,1)
print(X)
print(Y)
# Inicijalizacija i fitovanje modela
lm = LinearRegression()
model = lm.fit(X, Y)
#ispis parametara modela
print('alpha =',model.intercept_)
print('beta =', model.coef_)
#linearna funkcija y_model (regresirana)
y_model=lm.predict(X)
#dijagrami
plt.plot(X,y_model)
plt.scatter(X,Y,c='green')
plt.title('Linearna regresija y=alfa + beta X')
plt.show()
