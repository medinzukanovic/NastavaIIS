import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
df=pd.read_excel(r'C:\Users\HP\Desktop\dina.xlsx')
root= tk.Tk() #definisanje polaznog prozora
root.geometry("700x300")
root.title("Standard") #naziv polaznog prozora
pocetna=tk.Label(root, text="Dobrodošli") #ubacivanje texta
pocetna.pack() #inicijalizacija texta u dijaloškki okvir
a=df.iloc[0,1]
b=df.iloc[0,0]
def ponozi():
    c=a*b
    print(c)

print(a) 
#Tkinter tipka
def prikazi():
    plt.plot(df['a'],df['b'])
    plt.title('dina')
    plt.show()

tk.Button(root,text="pomnozi",command=ponozi).pack()
tk.Button(root,text="dijagram",command=prikazi).pack()
v = tk.StringVar()
unos=tk.Entry(root,textvariable=v).pack()
def prikazii():
    export_file_path = filedialog.asksaveasfilename(defaultextension='.xlsx')
    df[3,3]=v.get()
    df.to_excel(export_file_path, index = False, header=True)
    
tk.Button(root,text="dijagram",command=prikazii).pack()
#Kraj dijaloškog okvira

root.mainloop()