import serial
import time
import tkinter as tk
import matplotlib.pyplot as plt

root=tk.Tk()
root.geometry('200x200')
broj_stepova=0
granicnik=0
pret_unos=tk.StringVar()
pret_unos.set(0)
while (granicnik==0):
    print('Resetovanje graničnika')
    granicnik += 1

tk.Label(root,text='x: [mm]').pack()
unos=tk.StringVar()
unos.set(0)
tk.Entry(root,textvariable=unos).pack()
def naprijed():
    print(69)
    time.sleep(0.5)
    broj_stepova=int((float(unos.get())-float(pret_unos.get()))*200/(1.25))
    arduino=serial.Serial("COM5",9600)
    arduino.write(broj_stepova)
    arduino.close()
    time.sleep(0.5)
    print(1)
    print('prethodni unos')
    print(pret_unos.get())
    print('razlika')
    print(abs((float(pret_unos.get())-float(unos.get()))))
def nazad():
    print(69)
    time.sleep(0.5)
    broj_stepova=int((float(pret_unos.get())-float(unos.get()))*200/(1.25))
    print(broj_stepova)
    time.sleep(0.5)
    print(2)
    print('prethodni unos')
    print(pret_unos.get())
    print('razlika')
    print(abs((float(pret_unos.get())-float(unos.get()))))
def gas():
    print('Trenutni unos')
    print(unos.get())
    if(float(unos.get())==float(pret_unos.get())):
        print('Unijeli ste istu vrijednost')
        pret_unos.set(unos.get())
    if (float(unos.get())-float(pret_unos.get())>0):
        naprijed()
        print('naprijed')
        pret_unos.set(unos.get())
    if (float(unos.get())-float(pret_unos.get())<0):
        nazad()
        print('nazad')
        pret_unos.set(unos.get())
def hod300mm():
    granicnikk=0
    if (granicnikk!=0):
        print('Resetovanje graničnika')
        granicnikk += 1
    broj_stepova=int(300*200/(1.25))
    print(broj_stepova)
    print('naprijed')
    print('nazad')
tk.Button(root,text='Gas',command=gas).pack()
tk.Button(root,text='Hod 300mm',command=hod300mm).pack()

root.mainloop()