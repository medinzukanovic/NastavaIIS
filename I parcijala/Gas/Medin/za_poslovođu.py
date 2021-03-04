import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
import tkinter.messagebox
import matplotlib.pyplot as plt
import matplotlib
from openpyxl import Workbook
import mysql.connector
import email_sender
#Sql Import Plana
conn = mysql.connector.connect(host="localhost",port="3306",user="root",password="",database="standard")
cursor=conn.cursor()
df1 = pd.read_sql('SELECT * FROM `planirano`', con=conn)
conn.close()
###########################################
#Update svaki put kad klikne
def update_svakim_klikom():
    global df1
    conn = mysql.connector.connect(host="localhost",port="3306",user="root",password="",database="standard")
    cursor=conn.cursor()
    df1 = pd.read_sql('SELECT * FROM `planirano`', con=conn)
    conn.commit()
#Tkinter gui inicijalizacija
root= tk.Tk() #definisanje polaznog prozora
root.geometry("700x300")
root.title("Standard") #naziv polaznog prozora
pocetna=tk.Label(root, text="Dobrodošli") #ubacivanje texta
pocetna.pack() #inicijalizacija texta u dijaloškki okvir
######################################
#Import plana od jučer
lokacija=tk.filedialog.askopenfilename()
df = pd.read_excel(lokacija)
################################
#Tkinter inicijalizacija texta
pocetna1=tk.Label(root, text="Za svaku napravljenu stolicu kliknite na odgovarajući model")
pocetna1.pack()
#########################
#Varijable broj napravljenih, u startu je 0
broj_napravljenih_model1=0
broj_napravljenih_model2=0
broj_napravljenih_model3=0
########################
#Skladište od jučer
global broj_na_skaldistu_model1
global broj_na_skaldistu_model2
global broj_na_skaldistu_model3
#iloc[red,kolona]
#Deklaracija varijabli iz excel fajla
broj_na_skaldistu_model1=df.iloc[3,1]
broj_na_skaldistu_model2=df.iloc[3,2]
broj_na_skaldistu_model3=df.iloc[3,3]
#########################
#Koliko je planirano kupi iz SQL baze i dodjeljuje vrijednost
global planirano_model1
global planirano_model2
global planirano_model3
planirano_model1=df1.iloc[1,1]
planirano_model2=df1.iloc[1,2]
planirano_model3=df1.iloc[1,3]
########################
#Raćuna koliko fali
koliko_fali_model1= tk.IntVar()
koliko_fali_model1.set(planirano_model1-broj_na_skaldistu_model1)
koliko_fali_model2= tk.IntVar()
koliko_fali_model2.set(planirano_model2-broj_na_skaldistu_model2)
koliko_fali_model3= tk.IntVar()
koliko_fali_model3.set(planirano_model3-broj_na_skaldistu_model3)
##########################
#Funkcije kada se napravi model
def napravljen_model1():
    global broj_napravljenih_model1
    broj_napravljenih_model1+=1
    koliko_fali_model1.set(planirano_model1-broj_napravljenih_model1-broj_na_skaldistu_model1)
    update_svakim_klikom()
    
    
def napravljen_model2():
    global broj_napravljenih_model2
    broj_napravljenih_model2+=1
    koliko_fali_model2.set(planirano_model2-broj_napravljenih_model2-broj_na_skaldistu_model2)
    update_svakim_klikom()
   
    
def napravljen_model3():
    global broj_napravljenih_model3
    broj_napravljenih_model3+=1
    koliko_fali_model3.set(planirano_model3-broj_napravljenih_model3-broj_na_skaldistu_model3)
    update_svakim_klikom()
########################
#Tkinter Buttoni
tk.Button(root,text="Napravljen model 1",command=napravljen_model1).pack() #deklaracija tipke
tk.Button(root,text="Napravljen model 2",command=napravljen_model2).pack() #deklaracija tipke
tk.Button(root,text="Napravljen model 3",command=napravljen_model3).pack() #deklaracija tipke
###################################
#Presjek stanja grafik
def presjek_stanja():
    global top1
    
    global skladiste_sutra_model1
    if broj_napravljenih_model1+broj_na_skaldistu_model1>planirano_model1:
        skladiste_sutra_model1=broj_napravljenih_model1+broj_na_skaldistu_model1-planirano_model1
    else:
        skladiste_sutra_model1=0
    global skladiste_sutra_model2
    if broj_napravljenih_model2+broj_na_skaldistu_model2>planirano_model2:
        skladiste_sutra_model2=broj_napravljenih_model2+broj_na_skaldistu_model2-planirano_model2
    else:
        skladiste_sutra_model2=0
    global skladiste_sutra_model3
    if broj_napravljenih_model3+broj_na_skaldistu_model3>planirano_model3:
        skladiste_sutra_model3=broj_napravljenih_model3+broj_na_skaldistu_model3-planirano_model3
    else:
        skladiste_sutra_model3=0
    global skladiste1
    skladiste1=skladiste_sutra_model1
    global skladiste2
    skladiste2=skladiste_sutra_model2
    global skladiste3
    skladiste3=skladiste_sutra_model3
    labels = ['Skladiste jucer','Napravljeno danas','Skladiste sutra']
    model1 = [broj_na_skaldistu_model1,broj_napravljenih_model1,skladiste_sutra_model1]
    model2 = [broj_na_skaldistu_model2,broj_napravljenih_model2,skladiste_sutra_model2]
    model3 = [broj_na_skaldistu_model3,broj_napravljenih_model3,skladiste_sutra_model3]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/3, model1, width, label='Model 1')
    rects2 = ax.bar(x, model2, width, label='Model 2')
    rects3 = ax.bar(x + width/3, model3, width, label='Model 3')
    ax.set_ylabel('Komada')
    ax.set_title('Presjek stanja')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig('C:\\Users\\HP\Desktop\\Pokusaji\\zadnja.png')
    plt.show()
    

#Tkinter Button
tk.Button(root,text="Presjek stanja",command=presjek_stanja).pack() #deklaracija tipke
#######################
#Tkinter Labeli
tk.Label(root, textvariable=koliko_fali_model1).pack()
tk.Label(root, textvariable=koliko_fali_model2).pack()
tk.Label(root, textvariable=koliko_fali_model3).pack()

#Sačuvaj Excel za sutra
def sacuvaj_csv():
    
    df2 = pd.DataFrame(np.array([['Skladište na pocetku dana',broj_na_skaldistu_model1, broj_na_skaldistu_model2, broj_na_skaldistu_model3], ['Napravljenih',broj_napravljenih_model1, broj_napravljenih_model2, broj_napravljenih_model3], ['Isporučenih',planirano_model1, planirano_model2, planirano_model3],['Skladiste na kraju dana',broj_na_skaldistu_model1+broj_napravljenih_model1-planirano_model1,broj_na_skaldistu_model2+broj_napravljenih_model2-planirano_model2,broj_na_skaldistu_model3+broj_napravljenih_model3-planirano_model3]]),columns=['','Model 1', 'Model 2', 'Model 3'])
 
    
    export_file_path = filedialog.asksaveasfilename(defaultextension='.xlsx')
    df2.to_excel(export_file_path, index = False, header=True)
    conn = mysql.connector.connect(host="localhost",port="3306",user="root",password="",database="standard")
    cursor=conn.cursor()
    #Insert
    sql_insert = "INSERT INTO `presjek stanja`(`-`,`Model 1`, `Model 2`, `Model 3`, `Vrijeme`) VALUES (%s,%s,%s,%s,now())"
    val_insert = (['Skladište na pocetku dana',int(broj_na_skaldistu_model1),int(broj_na_skaldistu_model2),int(broj_na_skaldistu_model3)],['Napravljeno',int(broj_napravljenih_model1),int(broj_napravljenih_model2),int(broj_napravljenih_model3)],['Isporučeno',int(planirano_model1),int(planirano_model2), int(planirano_model3)],['Skladiste na kraju dana', int(broj_na_skaldistu_model1+broj_napravljenih_model1-planirano_model1),int(broj_na_skaldistu_model2+broj_napravljenih_model2-planirano_model2),int(broj_na_skaldistu_model3+broj_napravljenih_model3-planirano_model3)])
    cursor.executemany(sql_insert,val_insert)
    conn.commit()
    presjek_stanja()
    email_sender.salji_mail(export_file_path)
    

#Tkinter tipka
tk.Button(root,text="Sacuvaj CSV",command=sacuvaj_csv).pack()
#Kraj dijaloškog okvira
root.mainloop()