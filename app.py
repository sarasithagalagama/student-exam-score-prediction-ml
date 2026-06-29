import streamlit as st
import numpy as np
import joblib as jb
warnings.filterwarnings("ignore")


model = jb.load("notebooks/final_model.pkl")

st.title("Student Exam Score Prediction")

study_hours = st.slider("Study Hours", 0.0, 12.0, 2.0)
attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health Rating (1-10)", 0.0, 10.0, 5.0)
sleep_hours = st.slider("Sleep Hours per Day", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])

part_time_job_encoded = 1 if part_time_job == "Yes" else 0

if st.button("Predict Exam Score"):
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, part_time_job_encoded]])
    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))
    st.success(f"Predicted Exam Score: {prediction:.2f}")