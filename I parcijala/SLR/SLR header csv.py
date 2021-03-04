import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
#importovanje podataka
df=pd.read_csv(r'C:\Users\HP\Documents\Py Programs\SLR\data header.csv')
X=np.array(df['X']).reshape(-1,1)
Y=np.array(df['Y']).reshape(-1,1)
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
plt.scatter(X,Y,c='orange')
plt.title('Linearna regresija y=alfa + beta X')
plt.show()
