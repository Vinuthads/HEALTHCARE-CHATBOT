from tkinter import *
import mysql.connector
import os

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


def gg():
    print("hello")
    print(logidval.get())
    print(paswdval.get())
    try:
        cur = mydb.cursor()
        sql = "SELECT PWD FROM USR WHERE UID = %s"
        adr = (logidval.get(),)
        cur.execute(sql, adr)
        res = list(cur.fetchone())

        if paswdval.get() == res[0]:
            print("login successful")
            root.destroy()
            os.system('app.py')
        else:
            Label(root,text="invalid username or password",fg="red").grid(row=9,column=3)

    except :
        print("invalid username or password")
        Label(root,text="invalid username or password",fg="red").grid(row=9,column=3)

def hh():
    root.destroy()
    os.system('register.py')

bg = PhotoImage(file = "ima.png")
label1 = Label( root, image = bg)
label1.place(x=0,y=0)

head = Label(root,text="WELCOME!! TO HEALTHCARE CHAT-BOT",font=("times new roman",7))
head.place(x=20,y=10)

Label(root,text="LOGIN",font="comicsansms 13 bold",bg="violet red",pady=10,padx=70).place(x=200,y=40)
logid=Label(root,bg="grey",text="enter your username",font=("times new roman",13))
paswd=Label(root,bg="grey",text="enter the password",font=("times new roman",13))
logid.place(x=140,y=140)
paswd.place(x=140,y=200)

logidval=StringVar()
paswdval=StringVar()

loent=Entry(root,textvariable=logidval)
pent=Entry(root,show="*",textvariable=paswdval)

loent.place(x=340,y=140)
pent.place(x=340,y=200)
Button(text="submit",command=gg,pady=12,padx=10,bg="black",fg="white").place(x=300,y=250)
Button(text="Register",command=hh,pady=12,padx=10,bg="black",fg="white").place(x=400,y=250)

root.mainloop()