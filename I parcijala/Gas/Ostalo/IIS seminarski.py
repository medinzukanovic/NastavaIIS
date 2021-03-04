import numpy as np
from simple_pid import PID
import time
import matplotlib.pyplot as plt
import serial
import time
#O procesu
trajanje_procesa=500  #sekunde
interval_mjerenja=0.1 #sekunde
broj_mjerenja=trajanje_procesa/interval_mjerenja
#Parametri PID
Kp=1
Ki=0.3
Kd=0.05
#SetPoint
sp=30
zeljena_temperatura= np.zeros(int(broj_mjerenja)+1)  # set point
zeljena_temperatura[:] = sp
pid=PID(Kp, Ki, Kd, setpoint=sp)
#Limiti
pid.output_limits = (0, 10)
#pid.sample_time = 0.01
temp = np.zeros(int(broj_mjerenja)+1)
output = np.zeros(int(broj_mjerenja)+1)
ns=int(broj_mjerenja)
timee = np.linspace(0,ns/10,ns+1)
print(zeljena_temperatura)
pocetna_temp=20
i=0
ser=serial.Serial('COM5', 9600)
while i<broj_mjerenja:
    
    gas=float(ser.readline())
    plt.scatter(i,gas)
    plt.legend(['Temperatura'],loc='best')
    plt.ylabel('Temperatura')
    plt.ylim([-0.1,50])
    plt.xlabel('Vrijeme')
    plt.pause(0.1)
    #time.sleep(interval_mjerenja)
    i=i+1
plt.show()
    
