import mysql.connector
import tkinter as tk
#MySql konekcija
conn = mysql.connector.connect(host="localhost",port="3306",user="root",password="",database="standard")
#MySql kursor
cursor=conn.cursor()
#Variables
global m1_plan
global m2_plan
global m3_plan
#Insert
#sql_insert = "INSERT INTO `planirano`(`Model 1`, `Model 2`, `Model 3`, `Vrijeme`) VALUES (%s,%s,%s,now())"
#val_insert = (1,2,3)
#Update
#sql_update ="UPDATE `planirano` SET `Model 1`=%s,`Model 2`=%s,`Model 3`=%s,`Vrijeme`=now() WHERE 1"
#val_update = (5,4,6)
#Promjena Sql
#cursor.execute(sql_update,val_update)
#Tkinter vizualni
root= tk.Tk() #definisanje polaznog prozora
root.geometry("700x300")
root.title("Standard") #naziv polaznog prozora
pocetna=tk.Label(root, text="Dobrodošli") #ubacivanje texta
pocetna.pack() #inicijalizacija texta u dijaloškki okvir
pocetna1=tk.Label(root, text="Unesite planirani broj proizvoda") #ubacivanje texta
pocetna1.pack()
L1 = tk.Label(root, text="Model 1")
L1.pack()
E1 = tk.Entry(root, bd =5)
E1.pack()
L2 = tk.Label(root, text="Model 2")
L2.pack()
E2 = tk.Entry(root, bd =5)
E2.pack()
L3 = tk.Label(root, text="Model 3")
L3.pack()
E3 = tk.Entry(root, bd =5)
E3.pack()
def update_tabele():
    conn = mysql.connector.connect(host="localhost",port="3306",user="root",password="",database="standard")
    cursor=conn.cursor()
    m1_plan = int(E1.get())
    m2_plan = int(E2.get())
    m3_plan = int(E3.get())
    sql_update ="UPDATE `planirano` SET `Model 1`=%s,`Model 2`=%s,`Model 3`=%s,`Vrijeme`=now() WHERE 1"
    val_update = (m1_plan,m2_plan,m3_plan)
    cursor.execute(sql_update,val_update)
    conn.commit()
tk.Button(root,text='Update',command=update_tabele).pack()
conn.commit()
root.mainloop()