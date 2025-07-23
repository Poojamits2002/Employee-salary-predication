
import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("salary_predictor_model.pkl")

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("👔 Employee Salary Predictor")
st.markdown("Will this person earn more than 50K/year? Fill in the details below:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                                       'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                                       'Assoc-acdm', 'Assoc-voc', 'Doctorate', '5th-6th', '1st-4th'])
educational_num = st.slider("Education Number", 1, 16, 9)
marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced',
                                                 'Separated', 'Married-spouse-absent', 'Widowed'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                         'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                                         'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                                         'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                                             'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 10000, 0)
hours_per_week = st.slider("Hours per Week", 1, 99, 40)
native_country = st.selectbox("Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 
                                          'Canada', 'India', 'England', 'Cuba', 'Jamaica', 'Other'])

# Predict button
if st.button("Predict Salary"):
    input_df = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'fnlwgt': 1,
        'education': education,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }])

    pred = model.predict(input_df)[0]
    result = ">50K" if pred == 1 else "<=50K"
    st.success(f"💰 Predicted Income: **{result}**")
