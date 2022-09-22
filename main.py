# -*- coding: utf-8 -*-
"""
Created on Wed Jan 2 18:11:34 2021

@author: Sanjay
"""

from tkinter import *
from tkinter import ttk

import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import sqlite3
import json

l1=['Fever','Chills','Sweating','Headache','Vomiting','Muscle_pain','Rapid_breathing','Cough',
    'Diarrhea','Bloody_stools','Nausea','Fatigue','Yellow_skin','Kidney_failure',
    'Swelling_of_legs','Loss_of_appetite','Abdominal_pain','Dark_urine','Itching',
    'Light_coloured_stools','Rectal_Bleeding/pain','Burning_sensation_when_urinating',
    'Red_urine','Strong_smell_of_urine','Freuent_urination','Joint_pain','Redness_near_the_joint',
    'Swelling_near_the_joint','Warmth_near_the_joint','Difficulty_moving_the_joint','Stiffness',
    'Reduced_sense_of_smell','Thick_discharge_from_nose','Tooth_acne','Nasal_congestion',
    'Facial_pain','Ear_pain','Liquid_discharge_from_ear','Itching_around_the_ear',
    'Swelling_near_ear_canal','Redness_near_ears','Uncomfortable_moving_head',
    'Heavy_or_excessive_menstrual_bleeding','Prolonged_menstrual_periods',
    'Pelvic_pressure/pain','Pressure_on_the_lower_abdomen','Lower_Back_pain','Itching_near_anus','Sleeplessness','Worms_in_poo',
    'Pink_or_red_colouration_in_the_eyes','Itchy_eyes','Crust_in_eyes',
    'Burning_eyes','Blurred_vision','Sensitivity_to_light','Increased_tear_production',
    'A_soar_throat','Loss_of_voice/trouble_speaking','Need_to_clear_throat','Dry_cough',
    'Hoarseness','Grating_sensation','Increased_stiffness_in_the_joint',
    'Reduced_motion_of_the_joint','Swelling_near_the_joint','Runny_nose',
    'Chest_pain','Heartburn ','Excessive_saliva_in_the_mouth','Difficult_swallowing',
    'Regurgitation','Lump_near_throat','Coughing_up_blood','Swollen_neck','Dizziness',
    'Burning_sensation_near_veins','Skin_decoloration','Itchiness_near_veins',
    'Enlarged_veins','Pale_skin','Increased_hunger','Increased_thirst','Sores_that_donot_heal',
    'spinning_sensation','Ringing_in_the_ear','Hearing_loss','Sharppain_when_passing_stools','Rounded_shape_of_the_nail',
    'Swollen_finger_or_toe','Flaking_silver_patches_of_skin','Bloating',
    'Early_feeling_of_fullness','Headacne_backside','Stiffness_neck','Lack_of_coordination',
    'Numbness','Feeling_shaky','Increasing_pulse','Frequent_infections','Black_or_tarry_stools',
    'Pain_that_goes_off_after_eating','Soreness_around_patches_or_scalp','Thick_or_pitted_nails',
    'Yellowish_eyes','Changes_in_the_sensation','Difficulty_breathing']

disease=['Malaria','Jaundice','Hepatitis','Urinary Tract Infection','Gastroenteritis','Gout',
'Sinusitis','Otitis_externa','Fibroids','Threadworm','conjunctivities','Laryngitis',
'Osteoarthritis','Catarrh','Gerd','Tuberculosis','Varicose veins','Hypoglycemia','Pneumonia',
'Diabetes','Covid','Labyrinthitis','piles','Bronchiectasis','psoriasis','cholestasis','Anthrax',
'peptic ulcer','Cervical spondylosis']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)


df=pd.read_csv("Trainingg.csv")



df.replace({'Diseases':{'Malaria':0,'Jaundice':1,'Hepatitis':2,'Urinary Tract Infection':3,'Gastroenteritis':4,'Gout':5,
'Sinusitis':6,'Otitis externa':7,'Fibroids':8,'Threadworm':9,'conjunctivities':10,'Laryngitis':11,
'Osteoarthritis':12,'Catarrh':13,'Gerd':14,'Tuberculosis':15,'Varicose veins':16,'Hypoglycemia':17,
'Pneumonia':18,'Diabetes':19,'Covid':20,'Labyrinthitis':21,'piles':22,'Bronchiectasis':23,'psoriasis':24,'cholestasis':25,'Anthrax':26,
'peptic ulcer':27,'Cervical spondylosis':28}},inplace=True)

X= df[l1]

y = df[["Diseases"]]

tr=pd.read_csv("test.csv")

tr.replace({'Diseases':{'Malaria':0,'Jaundice':1,'Hepatitis':2,'Urinary Tract Infection':3,'Gastroenteritis':4,'Gout':5,
'Sinusitis':6,'Otitis externa':7,'Fibroids':8,'Threadworm':9,'conjunctivities':10,'Laryngitis':11,
'Osteoarthritis':12,'Catarrh':13,'Gerd':14,'Tuberculosis':15,'Varicose veins':16,'Hypoglycemia':17,
'Pneumonia':18,'Diabetes':19,'Covid':20,'Labyrinthitis':21,'piles':22,'Bronchiectasis':23,'psoriasis':24,'cholestasis':25,'Anthrax':26,
'peptic ulcer':27,'Cervical spondylosis':28}},inplace=True)


Xtest= tr[l1]

ytest = tr[["Diseases"]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

def message():
    
    if (len(NameEn.get())== 0  ):
        messagebox.showinfo("System","Kindly Fill the  Name")
    elif (len(str(ageen.get()))==0):
        messagebox.showinfo("System","Kindly Fill the  Age")
    elif (Symptom1.get() == "None" or Symptom2.get() == "None" or Symptom3.get() == "None" or Symptom4.get() == "None" or Symptom5.get() == "None" ):
        messagebox.showinfo("OPPS!!", "KINDLY ENTER  5 SYMPTOMS PLEASE")
    else :
        algorithm()

def algorithm():
    #NaiveBayes
    
    al1 = MultinomialNB()
    al1=al1.fit(X,np.ravel(y))
    a=al1
    
    y_pred=cross_val_predict(a, X, np.ravel(y), cv=3)
    print("NaiveBayess",end ="      ")
    print("%.3f" % accuracy_score(np.ravel(y), y_pred))
    
    #DecisionTree
    
    al2 = tree.DecisionTreeClassifier() 
    al2 = al2.fit(X,y)
    b=al2
    
    y_pred=cross_val_predict(b, X, np.ravel(y), cv=3)
    print("Decision Tree",end ="    ")
    print("%.3f" % accuracy_score(np.ravel(y), y_pred))
    
    #randomforest
    
    al3 = RandomForestClassifier(n_estimators=100)
    al3 = al3.fit(X,np.ravel(y))
    c=al3
    
    y_pred=cross_val_predict(c, X, np.ravel(y), cv=3)
    print("Randomforest",end ="     ")
    print("%.3f" % accuracy_score(np.ravel(y), y_pred))
    
    #knn
    
    al4=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    al4=al4.fit(X,np.ravel(y))
    d=al4
    
    y_pred=cross_val_predict(d, X, np.ravel(y), cv=3)
    print("KNN",end ="              ")
    print("%.3f" % accuracy_score(np.ravel(y), y_pred))
    
    #svm
    
    al5 = SVC(kernel='sigmoid',decision_function_shape='ovo')
    al5=al5.fit(X,np.ravel(y))
    e=al5
    
    y_pred=cross_val_predict(e, X, np.ravel(y), cv=3)
    print("svm",end ="              ")
    print("%.3f" % accuracy_score(np.ravel(y), y_pred))
    #logestic
    
    al6=LogisticRegression(multi_class='multinomial')
    al6=al6.fit(X,np.ravel(y))
    f=al6
    
    y_pred=cross_val_predict(f, X, np.ravel(y), cv=3)
    print("logestic",end ="         ")
    print("%.3f" % accuracy_score(np.ravel(y), y_pred))
    #mlp
    
    al7 = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    al7=al7.fit(X,np.ravel(y))
    g=al7
    
    y_pred=cross_val_predict(g, X, np.ravel(y), cv=3)
    print("MLP",end ="              ")
    print("%.3f" % accuracy_score(np.ravel(y), y_pred))
    
    
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    t1.delete("1.0", END)

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    #above are the complete algorithm part. GUI and further part are in use of Research.  Will be added one its published. 
