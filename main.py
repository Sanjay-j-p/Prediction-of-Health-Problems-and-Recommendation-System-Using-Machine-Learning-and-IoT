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

    inputtest = [l2]
    listt=[a,b,c,d,e,f,g]
    lst = []
    for pp in range(7):
        predict = list[pp].predict(inputtest)
        predicted=predict[0]
        
        

        h='no'
        for a in range(0,len(disease)):
            if(disease[predicted] == disease[a]):
                
                h='yes'
                break
        
        if (h=='yes'):
            
            lst.append(disease[a])
            t1.insert(END, str(pp+1)+"  "+disease[a]+"\n")
        
        else:
            t1.delete("1.0", END)
            t1.insert(END, "No Disease")
    
    word_counter = {}
    
    for word in lst:
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1
        popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
 
        top_3 = popular_words[:1]
        diseaseee=top_3
        t2.delete("1.0", END)
        t2.insert(END, top_3)
    #database
    
    conn = sqlite3.connect('database.db') 
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS database(Name StringVar,Age StringVar,Height StringVar,Weight StringVar,Blood_sugar StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 StringVar,Symtom5 StringVar,Disease StringVar)")
    c.execute("INSERT INTO database(Name,Age,Height,Weight,Blood_sugar,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?,?,?,?,?)",(NameEn.get(),ageen.get(),height.get(),weight.get(),blood_glucose.get(),Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),json.dumps(diseaseee)))
    conn.commit()  
    c.close() 
    conn.close()
def blood():
    if (len(str(blood_glucose.get()))== 0):
        messagebox.showinfo("System","Kindly Fill ")
    else:
        bb=int(blood_glucose.get())
        if bb>70:
            if bb<140:
                t1.delete("1.0", END)
                t1.insert(END, "Normal level \nif checked before eating")
            else:
                t1.delete("1.0", END)
                t1.insert(END, "high blood sugar (hyperglycemia) \nif checked before eating")
        else:
            t1.delete("1.0", END)
            t1.insert(END, "Low blood sugar (hypoglycemia) \nif checked before eating")
def bmi():
    if (len(str(height.get()))== 0 or len(str(weight.get()))== 0):
        messagebox.showinfo("System","Kindly Fill height and weight ")
    else:
        h=int(height.get())
        w=int(weight.get())
        bm=int(w/(h*0.01*0.01*h))
        t1.delete("1.0", END)
        t1.insert(END,bm)
        if bm>18:
            if bm<=25:
                t1.insert(END, "\nnormal")  
            elif 25<bm<30:
                t1.insert(END, "\nOverweight")
            elif 30<bm<35:
                t1.insert(END, "\nObesity-1")
            elif 35<bm<40:
                t1.insert(END, "\nObesity-2")
            else:
                t1.insert(END, "\nObesity-3")
        else:
            t1.insert(END, "\nThiness")
def Suggesions():
    
    my_dict = {'Malaria\n': 'If the person has diarrhea,high fever above 102F and other flu symptoms.It is better to consult the doctor less than 24hours',
               'Jaundice\n': 'Jaundice takes time to develop but it is noticed only when skin and white portion of eyes turn yellow with an abdominal discomfort.So,if patient notices in initial stages then he can follow few natural remedies like including  lemon,tomato,gingers,barley and mainly radishes in their diet and also need to contact the doctor',
               '{Urinary Tract Infection}\n':'UTI is noticed only when a person is having a sharp pain while peeing and urine colour is mainly cola or light blood colour and also there will be rectal pain. This not deadly but consulting doctor at earliest helps to recover fast. Natural remedies that can be followed are drinking a plenty of water,taking foods that consist of high Vitamin C sould be avoided ,avoid bladder irritants like alcohol,spicy food,caffeine and cranberry juice can help the person to recover faster',
               'Gastroenteritis\n':'Gastroentitis is noticed when a person have diarrhea ,stomach pain or abdominal cramps(mainly worse). The natural remedies that can be followed are having 1 tablespoon of raw apple cider vinegar with 1 cup of warm water,bowl of probiotic yogurt,1-2 teaspoons of raw honey with 1 cup water,An inch of cinnamon stick with 1 cup of water and honey as required,A cup of rice water,1 green or unripen banana,drinking a plenty of water always helps(mainly hot),getting ample rest and sleep well. This is not a deadly illness but visiting a doctor is always advisable',
               'piles\n':'Piles is noticed when a person finds red blood in poo(worse condition),itchy anus , pain around anus. Piles is not a deadly disease but consulting doctor is advisable and natural remedies that can be used to recover in initial stages(hammeroids) are drinking plenty of water , Organic radish juice , Green tea and Dried figs should be included in the diet.',
               'Pneumonia\n':'Pneumonia is not a deadly illness if we visit a doctor less than 10-12 days and it is mainly noticed when a person feels sharp pain on the chest , fever , coughing yellow , green or blood coloured mucus, difficult to breath. The natural remedies that can be followed and included in diet are Ginger,Vitamin C foods , Turmeric ,carrots , Honey , Dandelion tea , Steam Inhilation. Until complete recovery regular visiting of doctor is suggested',
               '{Varicose veins}\n':'Varicose veins are noticed when a person has swollen veins near legs,itchiness near veins and skin decolouration near veins. This is not a deadly illness and in initial stages it can be cured just by following natural remedies properly. Natural remedies are exercise regularly,drinking plenty of water,including cabbage,apple cider vinegar,Garlic in diet and also apply olive and vitamin E oil on to the legs near varicose veins and massage it',
               '{peptic ulcer}\n':'Peptic Ulcer is noticed when a person gets stomach pain in between meals , bloating , nausea or vomiting,along with tarry or dark stools(in worse condition). It is not a deadly illness but consulting doctor is compulsory. The natural remedies that should be followed are drinking plenty of water , eating more fiber content(vegetables and fruits),Honey ,yogurt and few probiotics and better to avoid coffee , chocolates , processed foods and following remedies make a person recover faster but cannot be cured with natural remedies alone.',
               'Hypoglycemia\n':'Hypoglycemia is related to lowering of blood sugar level . It is not easily noticeable . So , mainly sugar patients are effected with hypoglycemia. If it is noticed , it is better to consult doctor and if the person is sugar patient . The person need to have blood test frequently to know any abnormality in sugar levels. The Natural remedies that can be included are Licorice tea(the best option to replenish blood sugar levels),Herbal tea , avoid alcohol and drinks.',
               'Gout\n':'Gout is noticed when a person feels the intense joint pain(mainly during night or early morning) and swelling near the joint. This is not a deadly illness.  This occurs due to high uric acid buildup in blood. This can be cured with basic care and home remedies during initial stages.  So , Include tart cherry juice , warm water with apple cider vinegar , celery seeds , Milk thitle seeds , Bananas in the diet. It is also advisable to consult general practitioner or Rheumatologist. tea(the best option to replenish blood sugar levels),Herbal tea , avoid alcohol and drinks.',
               'Sinusitis\n':'Sinusitis is nasal illness , It is not a deadly illness but need to consult doctor regularly. It is noticed when there is nasal inflammation , mainly pain in the sinuses , nasal congestion and discharge along with headaches. The natural home remedies that can be followed are drink plenty of water , eat foods with antibacterial properties like ginger , garlic , onions and use eucalyptus oil to clear the sinuses , ease facial pain with warm compresses.',
               'Otitis_externa\n':'Otitis is an ear illness , It is not deadly but better to visit doctor as early as possible . It is mainly noticed when a person feels severe ear pain , swelling near ears or neck , face suddenly feels numb. The natural remedies used are helpful for a person to recover fast . They are cleaning and washing the ear area regularly, Lavender oil , eucalyptus oil are used to reduce the ear congestion and protect ears from bad weather condtions.',
               'Threadworm\n':'Threadworm is anus illness , It is not deadly illness . It can be noticed when the person spot worms in poo or near the back passage and feeling itchy all the time mainly during nights near anal area. It is always better to consult doctor. The natural home remedies used are application of coconut oil , eating raw garlic and carrots , drinking lemon juice, pine apple juice ,yogurt. These are used to help the person recover faster.',
               'conjunctivities\n':'Conjuctivities are eye related illness and it is not a deadly illness . It is noticed when a person’s eyes become pink or red , Increased tear production,  Discharge of pus or mucus. It is always better to consult doctor. The natural home remedies are apply aloe vera on eyelids , wash your eyes and hands regularly , drink vegetable and fruit juices.',
               'Bronchiectasis\n':'Bronchiectasis is lungs diseases and has got the symptoms of coughing up a lot of mucus(worse case bloody mucus),chest pain with harder to breath along with clubbing of nails , joint pains. It is better to consult a doctor less than 2-3 days depending on condition. The natural home remedies are mainly breathing exercises , regular body exercises and including ginger , vitamin c , bay leaf , onions in diet.',
               'Labyrinthitis\n':'Labyrinthitis is ear related infection, which is noticed when a person losses his balance , vertigo along with vomiting and nausea , loss of hearing. This is not a deadly disease , but visiting doctor is suggested and helpful. Natural home remedies are useful to recover fast , They are take plenty of rest , vertigo related exercises are recommended , heat salt and put in cloth and keep it near the ear , drink plenty of water , Lavendar oil can also be used like salt.',
               'Laryngitis\n':'Laryngitis is a throat related and respiratory infection , When a person swallows he gets a lot of pain , swollen neck with soar throat. Consulting doctor is advisable(should be consulted as fast as possible). The natural home remedies , That are to be implemented are rest your voice,  gargle warm salt water , apple cider vinegar , tea with honey , ginger root.',
               'Catarrh\n':'Catarrh is illness related to runny,  stuffy nose(thick mucus) , irritating cough,  Headache , loss of taste with mild hearing loss. It is better to consult a doctor. The natural home remedies that are to be followed are take sips of warm water , use a saline nasal rinse , Inhale steam , try oil pulling with coconut oil , gargle with salt water.',
               'Osteoarthritis\n':'Osteoarthristis is not a deadly disease , This is noticed when a person feels joint aches and soreness , bony enlargement in the middle and end joints of fingers,  stiffness after periods of rest. Consulting doctor is the best option like general practitioner(if it is not in worse condition) The natural home remedies that can be followed are hot and cold compresses , Epsom salt bath , Ginger , green tea , regular exercise and plenty of water , good diet.',
               'psoriasis\n':'Psoriasis is not a severe illness,this is noticed when a person gets swollen and tender joint,especially in a finger or toe,swollen leg,stiffness in morning that fades ways during the day. The patient need to consult the doctor regularly,until he recovers completely The natural remedies that can be followed are,Epsom salt bath,Oatmeal in diet,Apple cider vinegar,Ginger,Aloe vera,Standing under sun etc.',
               'Gerd\n':'Gerd is noticed when a person notices symptoms like burning sensation in the chest , difficulty swallowing , feeling of lump in the throat. This is not a deadly illness but , if neglected it can become worse. The natural remedies that are to be followed are regular exercise , drinking plenty of water , good diet with health weight,more salads to be included(fruits or vegetable).',
               'Anthrax\n':'Anthrax is noticed , when a person notices symptoms like Abdominal pain , loss of appetite , sore throat with difficult swallowing. Consulting doctor is advisable(as fast as possible) The natural remedies , that are to be followed are grape seed extract , ginger , neem leaves , oregano oil , Thyme oil should be drunk in a warm water.',
               'Cervical spondylosis\n':'Cervical is not a deadly illness ,but it is noticed when a person notices symptoms like lack of coordination during walking , numbness and weakness in the arms , hands , legs , feet ,Headaches that mostly occur at back of head. Consulting doctor is very very important and crutial. The natural home remedies that are to be followed are regular exercise , Epsom salt bath , Turmeric ,cayenne pepper , Cervical traction.',
               'Tuberculosis\n':'Tuberculosis is not a deadly illness but should be treated properly.  This can be noticed when a person notices symptoms like bad cough , chest pain , coughing up blood or mucus , chills , feeling weak and tired. Consulting doctor will help the person to recover fast. The Natural remedies that are to be followed are include garlic , Bananas , drumstick ,Indian gooseberry, Oranges , Custard apple , mint in diet.',
               'Fibroids\n':'Fibroids is not a deadly illness but should be treated as fast as possible . This is noticed when a person(female) have heavy menstrual bleeding , frequent urination , pelvic pain and pressure , backache and leg pains. Consulting doctor is advisable(as fast as possible). The natural remedies that are to be followed are including milk , apple cider vinegar , Garlic,  aloe vera , Lemon juice ,Green tea , onions , fish in diet',
               'Diabetes\n':'Diabetes is not a deadly illness but doctor should be consulted as fast as possible , else there is chance for any other organs getting effected . This can be noticed only through blood test but normal symptoms are increased hunger and thirst , blurry vision , sores that don’t heal ,fatigue , numbness in the feet or hands. The natural remedies that can be followed are including fruits in diet , having green tea , regular exercise , plenty of water , neem leaves etc.',
               'cholestasis\n':'Cholestasis can be noticed when a person have symptoms like Dark urine , yellowish skin and eyes , itchiness , pale stool. Doctor   should be consulted as fast as possible without any delay. The natural remedies that can be followed are Milk thistle(natural herb) , Vitamin k , d and calcium related foods should be taken , Dandelion tea.',
               'Hepatitis\n':'Hepatitis is not a deadly illness , this can be noticed when a person have symptoms like jaundice , general sense of unweel , headache , muscle pains , intense itching etc. Doctor should be consulted at the earliest. The natural remedies to be followed are tomatoes , ginger , garlic , water , mushrooms , Aloe vera should be included in the diet etc',
               'Covid\n':'Covid can be noticed , when a person notices high fever , dry cough , body aches , congestion , soar throat ,diarrhoea. If the person notices that in very early stages , self isolation and home treatment can help him to recover faster It is advisable to consult doctor. The natural home remedies that can help are drinking plenty of water,having a healthy diet,regular exercise (mainly lungs related exercises). If the person feels suffocation and chest pain then,It is advisable to admit in hospital It is a deadly disease,action should be taken within (24-36)hours'}
    abc=t2.get("1.0",END)
    #print(my_dict[abc])
    messagebox.showinfo("Suggesions",my_dict[abc])
def monitor():
    import os
    import webbrowser
    webbrowser.open('https://thingspeak.com/channels/1348347/private_show')
    os.startfile("C:\\Users\\sanjay\\Downloads\\iot_heartrate_monitor\\iot_heartrate_monitor.ino")
           
            
root = Tk()
root.title(" Health Prediction System from symptoms")
root.resizable(0,0)

Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)

w2 = Label(root, justify=LEFT, text=" Health Prediction System from symptoms ")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=0, columnspan=2, padx=100)



S1Lb = Label(root,  text="Symptom 1")
S1Lb.config(font=("Elephant", 15))
S1Lb.grid(row=9, column=0, pady=10)

S2Lb = Label(root,  text="Symptom 2")
S2Lb.config(font=("Elephant", 15))
S2Lb.grid(row=10, column=0, pady=10)

S3Lb = Label(root,  text="Symptom 3")
S3Lb.config(font=("Elephant", 15))
S3Lb.grid(row=11, column=0, pady=10)

S4Lb = Label(root,  text="Symptom 4")
S4Lb.config(font=("Elephant", 15))
S4Lb.grid(row=12, column=0, pady=10)

S5Lb = Label(root,  text="Symptom 5")
S5Lb.config(font=("Elephant", 15))
S5Lb.grid(row=13, column=0, pady=10)

lr = Button(root, text="Predict",height=2, width=10, command=message,fg ="red")
lr.config(font=("Elephant", 15))
lr.grid(row=21, column=0,padx=10)


lrr = Button(root, text="Monitor",height=2, width=10, command=monitor,fg ="red")
lrr.config(font=("Elephant", 15))
lrr.grid(row=12, column=2, padx=10,sticky=W)

bgg = Button(root, text="check \nblood \nsugar",height=2, width=10, command=blood)
bgg.config(font=("Elephant", 15))
bgg.grid(row=21, column=1,padx=10,sticky=W)


def Close():
    root.destroy()
    runfile('C:/Users/sanjay/Desktop/final year project/mineeeeeee/reviewfinal.py')
    
    
rs = Button(root, text="reset", command=Close,height=2,width=10)
rs.config(font=("Elephant", 15))
rs.grid(row=21,column=2,padx=10,sticky=W)


bmii = Button(root, text="BMI",height=3, width=10, command=bmi)
bmii.config(font=("Elephant", 15))
bmii.grid(row=21, column=1,padx=10)



b2 = Button(root, text = "Exit",height=2, width=10, command = root.destroy)
b2.config(font=("Elephant", 15))
b2.grid(row=21,column=2,padx=10)


OPTIONS = sorted(l1)


sym1 = ttk.Combobox(root, width = 35, textvariable = Symptom1) 
sym1['values'] = OPTIONS 

sym1.grid(row=9, column=1, pady=10)
sym1.current() 

sym2 = ttk.Combobox(root, width = 35, textvariable = Symptom2) 
sym2['values'] = OPTIONS 

sym2.grid(row=10, column=1, pady=10) 
sym2.current() 

sym3 = ttk.Combobox(root, width = 35, textvariable = Symptom3) 
sym3['values'] = OPTIONS 

sym3.grid(row=11, column=1, pady=10) 
sym3.current() 

sym4 = ttk.Combobox(root, width = 35, textvariable = Symptom4) 
sym4['values'] = OPTIONS 

sym4.grid(row=12, column=1, pady=10) 
sym4.current() 

sym5 = ttk.Combobox(root, width = 35, textvariable = Symptom5) 
sym5['values'] = OPTIONS 

sym5.grid(row=13, column=1, pady=10) 
sym5.current() 



Name = StringVar()

NameLb = Label(root, text="Name" )
NameLb.config(font=("Elephant", 15))
NameLb.grid(row=6, column=0, pady=20, sticky=W)

age = StringVar()

agelb = Label(root, text="Age")
agelb.config(font=("Elephant", 15))
agelb.grid(row=6, column=1, pady=20, sticky=W )

blood_glucose=StringVar()

bg = Label(root, text="Blood\nGlucose\n(mg/dL)")
bg.config(font=("Elephant", 15))
bg.grid(row=7, column=0, pady=20, sticky=W )

height=StringVar()

hg = Label(root, text="Height (cm)")
hg.config(font=("Elephant", 15))
hg.grid(row=7, column=1, pady=20,sticky=W)

weight=StringVar()
wg = Label(root, text="Weight (kg)")
wg.config(font=("Elephant", 15))
wg.grid(row=8, column=1, pady=20,sticky=W)



NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=0, pady=15)

ageen = Entry(root, textvariable=age)
ageen.grid(row=6, column=1, pady=15)

bgen = Entry(root, textvariable=blood_glucose)
bgen.grid(row=7, column=0, pady=15)

hgen = Entry(root, textvariable=height)
hgen.grid(row=7, column=1, pady=15)

wgen = Entry(root, textvariable=weight)
wgen.grid(row=8, column=1, pady=15)

res = Label(root, text="7 algorithms")
res.config(font=("Elephant", 15))
res.grid(row=6, column=2, pady=20, sticky=W )

t1 = Text(root, height=3, width=20)
t1.config(font=("Elephant", 20))
t1.grid(row=7, column=2, padx=20,sticky=W)

res1 = Label(root, text="END")
res1.config(font=("Elephant", 15))
res1.grid(row=8, column=2, pady=20, sticky=W )

b5 = Button(root, text = "Suggesions",height=2, width=10, command = Suggesions)
b5.config(font=("Elephant", 15))
b5.grid(row=10,column=2,padx=10,sticky=W)

t2 = Text(root, height=1, width=20)
t2.config(font=("Elephant", 20))
t2.grid(row=9, column=2, padx=10,sticky=W)


root.mainloop()
