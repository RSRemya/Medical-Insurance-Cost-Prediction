#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import pickle
import pandas as pd


# In[14]:


# Load the best trained model
model = pickle.load(open('insurance_charges_model.p', 'rb'))


# In[15]:


# Collect user inputs
age = st.number_input('Age', min_value=1, max_value=100, value=25)
st.write("Choose 0 for Male and 1 for Female")
sex = st.number_input('Sex', min_value=0, max_value=1)
bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
children = st.number_input('Children', min_value=0, max_value=5)
st.write("Choose 0 for Non-smoker and 1 for Smoker")
smoker = st.number_input('Smoker', min_value=0, max_value=1)
region_northeast = st.number_input('Region Northeast', min_value=0, max_value=1)
region_northwest = st.number_input('Region Northwest', min_value=0, max_value=1)
region_southeast = st.number_input('Region Southeast', min_value=0, max_value=1)
region_southwest = st.number_input('Region Southwest', min_value=0, max_value=1)

output = ""

if st.button("Predict"):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region_northeast, region_northwest, region_southeast, region_southwest]],
                              columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'])

    # Make prediction
    result = model.predict(input_data)
    st.success('The output of the above is {}'.format(result))


# In[ ]:




