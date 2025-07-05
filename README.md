# Cardiovascular Health Classifier 🫀

This project is a **machine learning model** that predicts an individual's cardiovascular health risk based on various health-related and demographic features. The model was developed using the **Random Forest Classifier** algorithm.

## 🚀 Project Overview

The goal of this project is to assist in early detection of cardiovascular health risks using readily available health indicators. This predictive model serves as a decision support tool in healthcare and wellness applications, deployed as a Streamlit web app for ease of use by nursing staff and clinicians.

> **🔔 Notice:** This model and application are open for improvement. Contributions, suggestions, and feedback are highly encouraged to enhance accuracy, usability, and overall performance. This is an ongoing project intended for educational and experimental purposes.
---

## 📊 Model Performance

### Confusion Matrix

![image](https://github.com/user-attachments/assets/fc1da2a8-8c1c-4e29-aebe-24e09c7caa81)

### Classification Report

![image](https://github.com/user-attachments/assets/1309d8cb-2d52-4b5a-8e55-a60ab31a51cb)

---

## 🧑‍⚕️ Features Used

The model takes the following features as input:

| Feature Name        | Description                                  |
|---------------------|----------------------------------------------|
| `BMI`               | Body Mass Index                              |
| `AgeCategory`       | Age group in years (mapped to midpoints, e.g., 21 for 18-24) |
| `Sex`               | Male or Female (encoded as 0/1)              |
| `Smoking`           | Smoker status (Yes/No, encoded as 0/1)       |
| `AlcoholDrinking`   | Alcohol consumption status (Yes/No, encoded as 0/1) |
| `Stroke`            | History of stroke (Yes/No, encoded as 0/1)   |
| `PhysicalHealth`    | Number of physically unhealthy days |
| `MentalHealth`      | Number of mentally unhealthy days |
| `DiffWalking`       | Difficulty walking (Yes/No, encoded as 0/1)  |
| `Diabetic`          | Diabetic status (Yes/No, encoded as 0/1)     |
| `PhysicalActivity`  | Engages in physical activity (Yes/No, encoded as 0/1) |
| `GenHealth`         | General health rating (1-5, Poor to Excellent) |
| `SleepTime`         | Average sleep time per day (hours)           |
| `Asthma`            | Asthma diagnosis (Yes/No, encoded as 0/1)    |
| `KidneyDisease`     | Kidney disease diagnosis (Yes/No, encoded as 0/1) |
| `SkinCancer`        | Skin cancer diagnosis (Yes/No, encoded as 0/1) |
| `Race_*`            | One-hot encoded race categories (e.g., Race_White, Race_Black) |

---

## 🛠 Technology Stack

- **Python 3.x**
- **scikit-learn** (Random Forest Classifier)
- **pandas** (Data manipulation)
- **joblib** (Model serialization)
- **Streamlit** (Interactive web app deployment)

---

## 📦 Model Deployment

The model is deployed as a **Streamlit web application**.

### 🔗 Live App:
👉 https://cardio-health-classifier.streamlit.app/
