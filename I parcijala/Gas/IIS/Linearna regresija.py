#Importovanje biblioteka
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#Importovanje podataka
df=pd.read_csv(r'C:\Users\HP\Documents\Py Programs\Podaci.csv')
X=np.array(df.iloc[:,0]).reshape(-1,1)
Y=np.array(df.iloc[:,1]).reshape(-1,1)
#Importovanje podataka
df=pd.read_csv(r'C:\Users\HP\Documents\Py Programs\Podaci.csv')
X=np.array(df.iloc[:,0]).reshape(-1,1)
Y=np.array(df.iloc[:,1]).reshape(-1,1)
# Inicijalizacija i fitovanje modela
lm = LinearRegression()
model = lm.fit(X, Y)
#Ispis parametara modela
print('alpha =',model.intercept_)
print('beta =', model.coef_)
print('beta =', model.score(X, Y, sample_weight=None))
