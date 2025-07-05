import streamlit as st
import joblib
import pandas as pd

model = joblib.load('cardio_health_classifier.pkl')
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

	# preprocess and encode the input data
	# Note: Ensure that the input data matches the expected format of the model
	# Note: The model expects specific feature names, so we need to ensure the input data matches those names.
	age_map = {'18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42,
				'45-49': 47, '50-54': 52, '55-59': 57, '60-64': 62, '65-69': 67,
				'70-74': 72, '75-79': 77, '80+': 82}
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
		'Race_American Indian/Alaskan Native': True if race == 'American Indian/Alaskan Native' else False,
		'Race_Asian': True if race == 'Asian' else False,
		'Race_Black': True if race == 'Black' else False,
		'Race_Hispanic': True if race == 'Hispanic' else False,
		'Race_Other': True if race == 'Other' else False,
		'Race_White': True if race == 'White' else False
	}


	input_df = pd.DataFrame([input_data])
	input_df = input_df[expected_features]	

	prediction = model.predict(input_df)[0]
	probability = model.predict_proba(input_df)[0][1]

	st.header("Prediction Result")
	if prediction == 1:
		st.error(f"Warning: The model predicts that you may have an increased risk of cardiovascular issues. Confidence: {probability * 100:.2f}%")
	else:
		st.success(f"Great! The model predicts that your cardiovascular health is good. Confidence: {(1 - probability) * 100:.2f}%")
