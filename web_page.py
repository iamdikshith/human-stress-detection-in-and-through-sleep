# for manipulations
import numpy as np
import pandas as pd

#for trained model
import pickle

#for deployment
import streamlit as st

#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

#for holding program (pause)
import time

#for loading image
from PIL import Image


# #---- TO SAVE THE MODEL ----
# #training a model
# from sklearn.tree import DecisionTreeClassifier
# dt_clf = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=5, min_samples_leaf=10, min_samples_split=10, min_weight_fraction_leaf=0.1)
# #fit the data - X and y
# dt_clf.fit(x_train, y_train)
# #dumping model through pickle
# filename = 'trained_model.sav'
# pickle.dump(dt_clf, open(filename,'wb'))
# #loading the saved model
# loaded_model = pickle.load(open('trained_model.sav','rb'))


#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))


#predictive model
def predictor(sr,rr,t,lm,bo,rem,sh,hr):
    #predict output
    prediction = loaded_model.predict((np.array([[sr,rr,t,lm,bo,rem,sh,hr]])))
    #return output
    return prediction


#title
st.title("Human Stress Detection in and through Sleep")
st.write('''
\n*Interactive Webpage*\n
'''
)


#take input
option = st.radio(
     "How would you like to give input?",
     ('Slider', 'Fill Values'))

if option == 'Fill Values':
    sr = st.number_input('Snoring Rate (dB)')
    rr = st.number_input('Respiration Rate (breaths per minute)')
    t = st.number_input('Body Temperature (◦F)')
    lm = st.number_input('Limb Movement')
    bo = st.number_input('Blood Oxygen')
    rem = st.number_input('Eye Movement')
    sh = st.number_input('Sleeping Hours (hr)')
    hr = st.number_input('Heart Rate (bpm)')
else:
    sr = st.slider('Snoring Rate (dB)', 45, 100, 71)
    rr = st.slider('Respiration Rate (breaths per minute)', 15, 30, 21)
    t = st.slider('Body Temperature (◦F)', 85, 110, 92)
    lm = st.slider('Limb Movement', 4, 20, 11)
    bo = st.slider('Blood Oxygen', 80, 100, 90)
    rem = st.slider('Eye Movement', 60, 105, 88)
    sh = st.slider('Sleeping Hours (hr)', 0, 12, 4)
    hr = st.slider('Heart Rate (bpm)', 50, 100, 64)


#to display the level for given stress
def level(n):
    if n==0:
        return "Low / Normal"
    elif n==1:
        return "Medium Low"
    elif n==2:
        return "Medium"
    elif n==3:
        return "Medium High"
    elif n==4:
        return "High"
    else:
        return ""


#Display final results
if st.button("RUN"):
    #pause the program for a while...
    with st.spinner(text='In progress'):
        time.sleep(2)
    #display output
    prediction=['']
    st.write("You have entered: ",sr,rr,t,lm,bo,rem,sh,hr)
    prediction=predictor(sr,rr,t,lm,bo,rem,sh,hr)
    st.write("Stress level is  ",prediction[0],"  ",level(int(prediction)))
    st.success('Stress level predicted!')


with st.expander("Other Info:"):
    # Display Features
    st.caption("Main Features:")
    st.code('''
sr - snoring rate
rr - respiration rate
t - body temperature
lm - limb movement
bo - blood oxygen
rem - eye movement
sh - sleeping hours
hr - heart rate
sl - stress level
    ''')
    # View dataset
    st.caption("First 5 rows:")
    df = pd.read_csv("stress.csv")
    st.write(df.head())
    #model statistics
    st.caption("Model used for learning:")
    st.code("DecisionTreeClassifier(max_depth=5, max_leaf_nodes=5, min_samples_leaf=10, min_samples_split=10, min_weight_fraction_leaf=0.1)")
    st.caption("Training accuracy:")
    st.write("0.998015873015873")
    st.caption("Accuracy:")
    st.write("0.9920634920634921")
    #Confusion matrix
    image = Image.open('CM.jpeg')
    st.image(image, caption='Confusion Matrix')
    #Decision Tree
    image = Image.open('DT.png')
    st.image(image, caption='Decision Tree')