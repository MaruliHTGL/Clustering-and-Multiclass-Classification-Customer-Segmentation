import streamlit as st
import numpy as np
import pandas as pd

# import ml package
import joblib
import os   

gen = {'Male': 0, 'Female': 1}
marital = {'Single': 0, 'Non-single (Divorced / Separated / Married / Widowed)': 1}
edu = {'Other / Unknown': 0, 'High School': 1, 'University': 2, 'Graduated': 3}
occu = {'Unemployed / Unskilled': 0, 'Skilled employee / Official': 1, 'Management / Self-employed / Highly Qualified Employee / Officer': 2}
settlement = {'Small City': 0, 'Mid-sized City': 1, 'Big City': 2}

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value
        
def load_scaler(scaler_file):
    loaded_scaler = joblib.load(open(os.path.join(scaler_file), 'rb'))
    return loaded_scaler
        
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

def run_ml_app():
    st.markdown("<h2 style = 'text-align: center;'> Input Your Customer Data </h2>", unsafe_allow_html=True)

    gender = st.radio('Gender', ['Male','Female'])
    marital_status = st.radio('Marital Status', ['Single','Non-single (Divorced / Separated / Married / Widowed)'])
    age = st.number_input("Age", 1, 100, value=25)
    education = st.selectbox("Education", ['High School', 'University', 'Graduated', 'Other / Unknown'])
    income = st.number_input("Income (USD)", 0, 999999, value=10000)
    occupation = st.selectbox("Occupation", ['Unemployed / Unskilled', 'Skilled employee / Official', 'Management / Self-employed / Highly Qualified Employee / Officer'])
    settlement_size = st.selectbox("Settlement Size", ['Small City', 'Mid-sized City', 'Big City'])

    result = {
            'gender': gender,
            'marital_status': marital_status,
            'age': age,
            'education': education,
            'income': income,
            'occupation': occupation,
            'settlement_size': settlement_size
    }
    
    df = pd.DataFrame(
        {
            'Gender': [gender],
            'Martial Status': [marital_status],
            'Age': [age],
            'Education': [education],
            'Income': [income],
            'Occupation': [occupation],
            'Settlement Size': [settlement_size]
        }
    )
    
    st.markdown("<h2 style = 'text-align: center;'>Your Customer Data </h2>", unsafe_allow_html=True)

    st.dataframe(df, height=50)

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in ['Male', 'Female']:
            res = get_value(i, gen)
            encoded_result.append(res)
        elif i in ['Single', 'Non-single (Divorced / Separated / Married / Widowed)']:
            res = get_value(i, marital)
            encoded_result.append(res)
        elif i in ['Other / Unknown', 'High School', 'University', 'Graduated']:
            res = get_value(i, edu)
            encoded_result.append(res)
        elif i in ['Unemployed / Unskilled', 'Skilled employee / Official', 'Management / Self-employed / Highly Qualified Employee / Officer']:
            res = get_value(i, occu)
            encoded_result.append(res)
        elif i in ['Small City', 'Mid-sized City', 'Big City']:
            res = get_value(i, settlement)
            encoded_result.append(res)

    single_array = np.array(encoded_result).reshape(1, -1)

    st.markdown("<h2 style = 'text-align: center;'> Prediction Result </h2>", unsafe_allow_html=True)

    scaling = load_scaler("scaler.pkl")    
    scaling_array = scaling.transform(single_array)

    model = load_model("model_dt.pkl")  
    prediction = model.predict(scaling_array)
    
    if prediction == 0:
        st.success("Cluster 0")
        st.write("Clusters with average customers of mature age, moderate income, working as skilled employees / officials, and living in mid-sized cities.")
    elif prediction == 1:
        st.success("Cluster 1")
        st.write("Clusters with average customers of old age, high income, working as management / self-employed / highly qualified employee / officer, and living in big cities.")
    elif prediction == 2:
        st.success("Cluster 2")
        st.write("Clusters with average customers of young age, small income, unemployed / unskilled, and living in small cities.")