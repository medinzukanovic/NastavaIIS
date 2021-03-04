import serial
import time
ser = serial.Serial('COM5', 9600) 
 
while True: 
 value = ser.readline() 
 print(value) 
 time.sleep(0.5) 