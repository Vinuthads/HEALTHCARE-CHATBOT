from tkinter import *
import mysql.connector
import os

from sklearn.utils import column_or_1d

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="vinu@4321",
  database="pro"
)
root = Tk()
root.title("MITRA")
root.geometry("600x400")
root.resizable(0, 0)

def bb():
    if paswdval.get()==cpaswdval.get() :


        try:
            cur = mydb.cursor()

            sql = "INSERT INTO USR (UID,PWD,PHNO,EMAIL) VALUES (%s, %s,%s,%s)"
            val=(logidval.get(),paswdval.get(),phoneval.get(),emailval.get())
            cur.execute(sql, val)
            mydb.commit()
            root.destroy()
            os.system('login.py')

        except:
            print("anger not good")
            Label(root,text="already exist").grid(row=8,column=5)
    
    else :
        Label(root,text="invalid").grid(row=7,column=5)

bg = PhotoImage(file = "ima.png")
label1 = Label( root, image = bg)
label1.place(x=0,y=0)

head = Label(root,text="WELCOME!! TO HEALTHCARE CHAT-BOT",font=("times new roman",7))
head.place(x=20,y=10)

Label(root,text="REGISTER",font="comicsansms 13 bold",bg="violet red",pady=10,padx=70).place(x=250,y=50)
logid=Label(root,text="enter  username",bg="grey")
paswd=Label(root,text="enter  password",bg="grey")
cpaswd=Label(root,text="confirm  password",bg="grey")
phno=Label(root,text="enter phone no",bg="grey")
email=Label(root,text="enter  email",bg="grey")
logid.place(x=100,y=140)
paswd.place(x=100,y=170)
cpaswd.place(x=100,y=200)
phno.place(x=100,y=230)
email.place(x=100,y=260)

logidval=StringVar()
paswdval=StringVar()
cpaswdval=StringVar()
phoneval=StringVar()
emailval=StringVar()
loent=Entry(root,textvariable=logidval)
pent=Entry(root,show="*",textvariable=paswdval)
cpent=Entry(root,textvariable=cpaswdval)
phent=Entry(root,textvariable=phoneval)
emaent=Entry(root,textvariable=emailval)
loent.place(x=230,y=140)
pent.place(x=230,y=170)
cpent.place(x=230,y=200)
phent.place(x=230,y=230)
emaent.place(x=230,y=260)
Button(text="submit",command=bb,pady=12,padx=14,bg="black",fg="white").place(x=330,y=300)

root.mainloop()