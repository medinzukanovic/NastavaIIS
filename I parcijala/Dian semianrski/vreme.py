import time
import serial
import numpy as np
import pandas as pd
gas=['prvi']
ser=serial.Serial('COM5', 9600)
while True:
    gas.append(ser.read())
    print(gas)