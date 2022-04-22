import nltk
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import datetime
from cProfile import label
import time
import random

"""dd()"""
import tkinter as tk
import tkinter.simpledialog as simpledialog
from tkinter import messagebox

ROOT = tk.Tk()
ROOT.withdraw()

warnings.filterwarnings("ignore", category=DeprecationWarning)
global disease_input
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

global inp

conf_inp = 0
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
cols = training.columns
cols = cols[:-1]
print(cols)
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

# mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print(scores.mean())

model = SVC()
model.fit(x_train, y_train)
print("for svm: ")
print(model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


def readn(nstr):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}




def kj():
    return 0




for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if ((sum * days) / (len(exp) + 1) > 13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)





def check_pattern(dis_list, inp):
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:

        
        if regexp.search(item):
            pred_list.append(item)
            
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, item


def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {}

    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return disease


def tree_to_code(tree, feature_names, disease_input, num_days):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    # conf_inp=int()
    while True:

        # link
        conf, cnf_dis = check_pattern(chk_dis, disease_input)

        if conf == 1:
            print("searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = kj()
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
            

        else:
            print("Enter valid symptom.")
            readn("enter valid symptom")
            USER_INP1 = simpledialog.askstring(title="MITRA", prompt="symptom")
            readn("from how many days are you experiencing " + USER_INP1)
            USER_INP2 = simpledialog.askstring(title="MITRA", prompt="no of days")
            tree_to_code(clf, cols, USER_INP1, int(USER_INP2))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("Are you experiencing any ")
            readn("Are you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                readn("Are you experiencing any "+syms)
                USER_INP = simpledialog.askstring(title="Test", prompt=syms)

                print(syms, "? : ", end='')

                inp = str(syms)
                while True:
                    inp = USER_INP
                    print(USER_INP)

                    if (inp == "yes" or inp == "no"):
                        break
                    else:
                        readn("provide proper answers that is yes or no ")
                        USER_INP = simpledialog.askstring(title="Test", prompt=syms)
                if (inp == "yes"):
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)

            if (present_disease[0] == second_prediction[0]):
                print("You may have ", present_disease[0])
                messagebox.showinfo("you may have", present_disease[0])
                readn("you may have"+ present_disease[0])

                print(description_list[present_disease[0]])
                messagebox.showinfo("you may have", description_list[present_disease[0]])
                readn(description_list[present_disease[0]])
                

            else:
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])
                readn("you may have"+ present_disease[0])
                readn("you may have"+ description_list[present_disease[0]])
                readn("you may have"+ description_list[second_prediction[0]])
                messagebox.showinfo("you may have", present_disease[0])
                messagebox.showinfo("you may have", description_list[present_disease[0]])
                messagebox.showinfo("you may have", description_list[second_prediction[0]])
            precution_list = precautionDictionary[present_disease[0]]
            readn("Take following measures")
            print("Take following measures : ")
            

            for i, j in enumerate(precution_list):
                print(i + 1, ")", j)
                readn(j)
                messagebox.showinfo("Take following measures ", str(i)+")" + str(j))

           
    recurse(0, 1)


getSeverityDict()
getDescription()
getprecautionDict()


def wish():
    hr=int(datetime.datetime.now().hour)
    if hr>=0 and hr<12:
        readn("good morning")
        print("good morning")
    if hr >= 12 and hr < 18:
        readn("good afternoon")
        print("good afternoon")
    else :
        readn("good evening")
        print("good evening")

wish()


readn("your health assistant here")
readn("please enter the symptoms you are experiencing")
USER_INP1 = simpledialog.askstring(title="MITRA", prompt="symptom")
readn("from how many days are you experiencing "+USER_INP1)
USER_INP2 = simpledialog.askstring(title="MITRA", prompt="no of days")
tree_to_code(clf, cols, USER_INP1, int(USER_INP2))
readn("take care ")
