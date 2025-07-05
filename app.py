import streamlit as st
import joblib
import pandas as pd
import requests
import os

MODEL_URL = 'https://huggingface.co/spaces/marckieee/models/resolve/main/cardio_health_classifier.pkl'
MODEL_FILE = 'cardio_health_classifier.pkl'

def download_model(url=MODEL_URL, output=MODEL_FILE):
    if not os.path.exists(output):
        try:
            st.info("Downloading model, please wait...")
            response = requests.get(url)
            response.raise_for_status()  # Raise error if bad status
            with open(output, 'wb') as f:
                f.write(response.content)
            st.success(f"Model downloaded successfully. File size: {os.path.getsize(output) / 1024:.2f} KB")
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            st.stop()

@st.cache_resource(show_spinner="Loading model...")
def load_model(file_path=MODEL_FILE):
    model = joblib.load(file_path)
    return model

download_model()
model = load_model()
expected_features = model.feature_names_in_

st.markdown("<h1 style='text-align: center;'>Cardiovascular Health Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 30px;'>This app predicts the cardiovascular health of an individual based on their input data.</p>", unsafe_allow_html=True)

with st.form('patient_data'):
    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        bmi = st.number_input('Body Mass Index (BMI)', min_value=10.0, max_value=60.0, step=0.1)
        age_category = st.selectbox('Age Category', ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'])
        sex = st.selectbox('Sex', ['Male', 'Female'])
        smoking = st.selectbox('Smoking', ['Yes', 'No'])
        alcohol_drinking = st.selectbox('Alcohol Drinking', ['Yes', 'No'])
        stroke = st.selectbox('Stroke', ['Yes', 'No'])
        physical_health = st.number_input('Physical Health (days)', min_value=0, max_value=30, step=1)
        mental_health = st.number_input('Mental Health (days)', min_value=0, max_value=30, step=1)

    with col2:
        race = st.selectbox('Race', ['White', 'Black', 'Asian', 'Hispanic', 'American Indian/Alaskan Native', 'Other'])
        diabetic = st.selectbox('Diabetic', ['Yes', 'No'])
        physical_activity = st.selectbox('Physical Activity', ['Yes', 'No'])
        general_health = st.selectbox('General Health', ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        sleep_time = st.number_input('Sleep Time (hours)', min_value=0, max_value=24, step=1)
        asthma = st.selectbox('Asthma', ['Yes', 'No'])
        kidney_disease = st.selectbox('Kidney Disease', ['Yes', 'No'])
        skin_cancer = st.selectbox('Skin Cancer', ['Yes', 'No'])

    diff_walking = st.selectbox('Difficulty Walking', ['Yes', 'No'])

    st.markdown("<br/>", unsafe_allow_html=True)
    submit = st.form_submit_button('Predict', use_container_width=True)
    st.markdown("<br/>", unsafe_allow_html=True)

if submit:
    age_map = {
        '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42,
        '45-49': 47, '50-54': 52, '55-59': 57, '60-64': 62, '65-69': 67,
        '70-74': 72, '75-79': 77, '80+': 82
    }
    health_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Very Good': 4, 'Excellent': 5}

    input_data = {
        'BMI': bmi,
        'AgeCategory': age_map[age_category],
        'Sex': 1 if sex == 'Male' else 0,
        'Smoking': 1 if smoking == 'Yes' else 0,
        'AlcoholDrinking': 1 if alcohol_drinking == 'Yes' else 0,
        'Stroke': 1 if stroke == 'Yes' else 0,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'DiffWalking': 1 if diff_walking == 'Yes' else 0,
        'Diabetic': 1 if diabetic == 'Yes' else 0,
        'PhysicalActivity': 1 if physical_activity == 'Yes' else 0,
        'GenHealth': health_map[general_health],
        'SleepTime': sleep_time,
        'Asthma': 1 if asthma == 'Yes' else 0,
        'KidneyDisease': 1 if kidney_disease == 'Yes' else 0,
        'SkinCancer': 1 if skin_cancer == 'Yes' else 0,
        'Race_American Indian/Alaskan Native': race == 'American Indian/Alaskan Native',
        'Race_Asian': race == 'Asian',
        'Race_Black': race == 'Black',
        'Race_Hispanic': race == 'Hispanic',
        'Race_Other': race == 'Other',
        'Race_White': race == 'White'
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[expected_features] 

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.header("Prediction Result")
    if prediction == 1:
        st.error(f"The model predicts increased cardiovascular risk.\n\nConfidence: {probability * 100:.2f}%")
    else:
        st.success(f"The model predicts good cardiovascular health.\n\nConfidence: {(1 - probability) * 100:.2f}%")
