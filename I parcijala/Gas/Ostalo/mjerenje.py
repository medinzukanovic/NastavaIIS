import serial
import time
import numpy as np
import pandas as pd
ser=serial.Serial('COM5', 9600)
niz=np.zeros(50)
for i in range(0,50,1):
     value =float(ser.readline())
     niz[i]=(value)
     time.sleep(1)
df=pd.DataFrame(niz)
df.to_csv(r'C:\Users\HP\Desktop\Podaci.csv')
